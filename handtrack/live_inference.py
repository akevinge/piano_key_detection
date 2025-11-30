import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import pygame  # For audio playback
import threading
import time

import mediapipe as mp
from midi import PseudoLandmarkList, FINGER_TIPS, PianoEvent
from piano_segmentation import PianoSegmentationModel, generate_piano_bbs_88_key

from train import (
    PianoPressNet,
    ACTIVE_FEATURES,
    collate_skip_erroneous,
    PianoPressDataset,
    load_data,
    load_metadata,
    extract_features,
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CONTEXT_LENGTH = 21
LABEL_MAP = {"no_event": 0, "pressed": 1, "released": 2}
REVERSE_LABEL_MAP = {0: "no_event", 1: "pressed", 2: "released"}


# ============================================================================
# AUDIO SETUP
# ============================================================================
pygame.mixer.init()


# Audio playback in separate thread to avoid blocking
class AudioPlayer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.is_playing = False

    def play(self):
        if self.audio_path:
            try:
                pygame.mixer.music.load(self.audio_path)
                pygame.mixer.music.play()
                self.is_playing = True
            except Exception as e:
                print(f"Error loading audio: {e}")

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False

    def pause(self):
        pygame.mixer.music.pause()

    def unpause(self):
        pygame.mixer.music.unpause()


# ============================================================================
# LOAD MODEL AND RUN INFERENCE
# ============================================================================

print("Loading evaluation data...")
eval_path = "data/have-yourself-a-merry-little-christmas/events.json"
video_path = "data/have-yourself-a-merry-little-christmas/video.mp4"
hands_path = "data/have-yourself-a-merry-little-christmas/hand_data.json"
audio_path = (
    "data/have-yourself-a-merry-little-christmas/audio.mp3"  # Add your audio file path
)

eval_data = load_data([eval_path])
eval_dataset = PianoPressDataset(eval_data, ACTIVE_FEATURES)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_skip_erroneous
)

print(f"Loaded {len(eval_dataset)} evaluation samples")

# Initialize model
print("\nInitializing model...")
model = PianoPressNet(
    num_features=len(ACTIVE_FEATURES), seq_len=CONTEXT_LENGTH, num_classes=3
)

# Load trained weights
print("Loading model weights from 'models/handtrack_piano_press_net.pth'...")
model.load_state_dict(torch.load("models/handtrack_piano_press_net.pth"))
model.eval()

with open(hands_path, "r") as f:
    hands_data = json.load(f)

frames_with_events = {ev.frame_index: ev for ev in eval_data}

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay in milliseconds between frames

key_bbs = load_metadata(eval_path)["key_bbs"]

# Initialize audio player
audio_player = AudioPlayer(audio_path)

print("\nControls:")
print("- SPACE: Play/Pause")
print("- Q: Quit")
print("\nStarting playback in 3 seconds...")
time.sleep(3)

# Start audio playback
audio_player.play()

frame_i = 0
paused = False
start_time = time.time()
pause_time = 0

while cap.isOpened():
    if not paused:
        # Calculate expected frame based on elapsed time
        elapsed_time = time.time() - start_time - pause_time
        expected_frame = int(elapsed_time * fps)

        # Skip frames if we're behind, or wait if we're ahead
        if frame_i < expected_frame:
            # We're behind, skip frames
            while frame_i < expected_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_i += 1
            if not ret:
                break
        elif frame_i > expected_frame:
            # We're ahead, wait a bit
            time.sleep((frame_i - expected_frame) / fps)

        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w, _ = frame.shape
        hands = hands_data[frame_i]
        context = hands_data[max(0, frame_i - 10) : min(len(hands_data), frame_i + 11)]

        # Make each key white
        for x1, y1, x2, y2 in key_bbs.values():
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

        if len(context) == 21:
            for hand_i, hand in enumerate(hands):
                mp_draw.draw_landmarks(
                    frame,
                    PseudoLandmarkList(hand["landmarks"]),
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 255),
                        thickness=2,
                        circle_radius=2,
                    ),
                    mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                )
                for fingertip in FINGER_TIPS:
                    lm = hand["landmarks"][fingertip]
                    # Check which key this lm is closest to
                    x, y = lm["x"], lm["y"]
                    px, py = int(x * img_w), int(y * img_h)
                    closest_key = None
                    min_dist = float("inf")
                    for key, (x1, y1, x2, y2) in key_bbs.items():
                        key_center_x = (x1 + x2) // 2
                        dist = abs(px - key_center_x)
                        if dist < min_dist:
                            min_dist = dist
                            closest_key = key

                    if closest_key is not None and min_dist < 100:
                        ev = PianoEvent(
                            note=key,
                            velocity=None,
                            timestamp=frame_i / fps,
                            event_type=None,
                            hand_used=hand_i,
                            finger_used=fingertip,
                            frame_index=frame_i,
                            hand_context=context,
                            midi_time=None,
                            key_bb=key_bbs[closest_key],
                        )
                        # Do inference
                        tensors = extract_features(ev, ACTIVE_FEATURES)
                        if tensors is None:
                            continue
                        input_tensor, _ = tensors
                        input_tensor = input_tensor.unsqueeze(0)
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            _, predicted = torch.max(output.data, 1)
                            pred_label = REVERSE_LABEL_MAP[predicted.item()]
                            confidence = probabilities[0][predicted.item()] * 100
                            if pred_label != "no_event" and confidence > 0.7:
                                x1, y1, x2, y2 = key_bbs[closest_key]
                                cv2.rectangle(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (
                                        (0, 0, 255)
                                        if pred_label == "pressed"
                                        else (0, 255, 0)
                                    ),
                                    2,
                                )

        cv2.imshow("Piano Hand Tracking Inference", frame)
        frame_i += 1

    # Handle keyboard input with minimal delay
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Press 'q' to quit
        break
    elif key == ord(" "):  # Press SPACE to pause/unpause
        if not paused:
            # Entering pause
            pause_start = time.time()
            audio_player.pause()
        else:
            # Exiting pause
            pause_time += time.time() - pause_start
            audio_player.unpause()
        paused = not paused

# Cleanup
audio_player.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
