import os
import math
import mido
import cv2
import json
import random
from tqdm import tqdm
import mediapipe as mp
from dataclasses import dataclass
from typing import Literal

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from handtrack import PianoHandTracker
from piano_segmentation import PianoSegmentationModel, generate_piano_bbs_88_key

FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]


@dataclass
class PseudoLandmark:
    x: float
    y: float
    z: float

    def HasField(self, _):
        # The drawing function checks 'visibility' and 'presence'.
        # We return False to tell it "we don't have this data, just draw it."
        return False


class PseudoLandmarkList:
    def __init__(self, landmarks_list):
        # Converts a list of dicts [{'x':0...}, ..] into a list of objects
        self.landmark = [PseudoLandmark(**lm) for lm in landmarks_list]


def serialize_hand_frame(frame_result):
    """
    Converts a single frame of MediaPipe results into a JSON-serializable list.
    Preserves Handedness (Left/Right) and Confidence scores.
    """
    if frame_result is None:
        return None

    # Check for landmarks
    if (
        not hasattr(frame_result, "multi_hand_landmarks")
        or not frame_result.multi_hand_landmarks
    ):
        return []

    # specific check for handedness just in case
    if (
        not hasattr(frame_result, "multi_handedness")
        or not frame_result.multi_handedness
    ):
        # Fallback if landmarks exist but handedness is missing (rare)
        return []

    serialized_hands = []

    # Use zip to iterate over landmarks and handedness simultaneously
    for hand_landmarks, handedness in zip(
        frame_result.multi_hand_landmarks, frame_result.multi_handedness
    ):

        # Access the classification (Left/Right)
        # handedness.classification is a list, usually containing just one item
        classification = handedness.classification[0]
        label = (
            "Left" if classification.label == "Right" else "Right"
        )  # Flip labels due to camera mirroring
        score = classification.score  # Probability (0.0 to 1.0)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(
                {
                    "x": round(lm.x, 5),
                    "y": round(lm.y, 5),
                    "z": round(lm.z, 5),
                }
            )

        # Structure the object to include metadata
        serialized_hands.append(
            {
                "type": label,  # "Left" or "Right"
                "score": round(score, 4),  # Confidence
                "landmarks": landmarks,
            }
        )

    return serialized_hands


class PianoEvent:
    def __init__(
        self,
        note,
        velocity,
        timestamp,
        event_type,
        hand_used=None,
        finger_used=None,
        frame_index=0,
        hand_context=None,
        midi_time=None,
        key_bb=None,
    ):
        self.note = note
        self.velocity = velocity
        self.timestamp = timestamp
        # event_type: 'pressed' or 'released'
        self.event_type = event_type

        # Calculated later
        self.hand_used = hand_used
        self.finger_used = finger_used
        self.frame_index = frame_index
        self.hand_context = hand_context  # The 21-frame window
        # Time in the MIDI file (not aligned to video)
        # Only pressed/released events have this
        self.midi_time = midi_time
        self.key_bb = key_bb

    def __repr__(self):
        return f"<Event Note: {self.note} ({self.event_type}) ({self.hand_used}) ({self.finger_used}) @ {self.timestamp:.2f}s>"

    def has_finger_key_matched(self) -> bool:
        """Returns True if both hand_used and finger_used are set."""
        return self.hand_used is not None and self.finger_used is not None

    def get_used_hand_landmarks(self, offset=0):
        """Returns the hand frame at the specified offset in the context window."""
        if (
            self.hand_context is None
            or self.hand_used is None
            or offset not in range(len(self.hand_context))
            or len(self.hand_context[offset])
            < self.hand_used + 1  # E.g. 1 hand but asked for 2nd (index 1)
        ):
            return None

        return self.hand_context[offset][self.hand_used]["landmarks"]

    def get_used_fingertip_landmark(self, offset=0):
        """Returns the hand frame at the specified offset in the context window."""
        if (
            self.hand_context is None
            or self.hand_used is None
            or self.finger_used is None
        ):
            return None

        return self.hand_context[offset][self.hand_used]["landmarks"][self.finger_used]

    def to_dict(self):
        """Prepares this object for JSON serialization."""
        return {
            "note": self.note,
            "velocity": self.velocity,
            "timestamp": round(self.timestamp, 4),
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            # Serialize the window of 21 frames
            "hand_context": self.hand_context,
            "finger_used": self.finger_used,
            "hand_used": self.hand_used,
            "midi_time": self.midi_time,
            "key_bb": self.key_bb,
        }


def process_and_save(
    piano_seg_model_path: str,
    midi_path: str,
    video_path: str,
    hand_output_path: str,
    output_json_path: str,
    first_note_press_frame=0,
    black_width=20,
    black_height=150,
    debug=True,
):
    # 1. Video Metadata
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        raise ValueError("Could not read video FPS.")

    # Attempt to get bounding box of piano from middle frame.
    mid_frame_index = total_frames // 2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
    ret, mid_frame = cap.read()
    if not ret:
        raise ValueError("Could not read middle frame for piano segmentation.")
    cap.release()

    piano_seg_model = PianoSegmentationModel(piano_seg_model_path)
    piano_bb = piano_seg_model.segment_piano(mid_frame)
    if piano_bb is None:
        raise ValueError("Could not detect piano in the video.")

    key_bbs = generate_piano_bbs_88_key(
        width=mid_frame.shape[1],
        height=mid_frame.shape[0],
        padding=(
            piano_bb[1],
            mid_frame.shape[1] - piano_bb[2],
            mid_frame.shape[0] - piano_bb[3],
            piano_bb[0],
        ),
        black_width=black_width,
        black_height=black_height,
    )
    if debug:
        # Draw piano and keys on mid_frame for debugging
        debug_frame = mid_frame.copy()
        x1, y1, x2, y2 = piano_bb
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kx1, ky1, kx2, ky2 in key_bbs.values():
            cv2.rectangle(debug_frame, (kx1, ky1), (kx2, ky2), (255, 0, 0), 1)
        cv2.imshow("Piano and Keys", debug_frame)
        # Wait for Q
        while cv2.waitKey(100000) != ord("q"):
            pass
        cv2.destroyAllWindows()

    # Ensure that there is only 1 trac
    mid = mido.MidiFile(midi_path)
    total_time = mid.length
    assert math.isclose(
        total_time, total_frames / fps, abs_tol=10
    ), f"MIDI duration ({total_time}s) differs from video by >=10s ({total_frames/fps}s)."

    # 2. Run Tracker
    if not os.path.exists(hand_output_path):
        print("1. Processing video (this may take a while)...")
        tracker = PianoHandTracker(video_path)
        hands_data = tracker.process_video(show=False)
        hands_data = [serialize_hand_frame(frame) for frame in hands_data]

        with open(hand_output_path, "w") as f:
            json.dump(hands_data, f)
    else:
        print("1. Loading existing hand tracking data...")
        with open(hand_output_path, "r") as f:
            hands_data = json.load(f)

    assert math.isclose(
        len(hands_data), total_frames, abs_tol=10
    ), f"Hand data frames ({len(hands_data)}) differs from video frames ({total_frames}) by >=10."

    # 3. Parse MIDI
    print("2. Parsing MIDI and aligning events...")
    mid = mido.MidiFile(midi_path)
    events = []
    current_time = 0.0
    total_time = mid.length
    # Get the first timestamp in iter(mid) without consuming the iterator
    mid_iter = [msg for msg in mid]
    # Find time of first message with key press
    first_note_time = 0.0
    for msg in mid_iter:
        if msg.type == "note_on" and msg.velocity > 0:
            break
        first_note_time += msg.time

    first_video_note_time = first_note_press_frame / fps
    video_time_offset = first_video_note_time - first_note_time
    print(first_note_time, first_video_note_time)
    print(video_time_offset, "s offset between MIDI and video")

    with tqdm(total=total_time, desc="Processing MIDI") as pbar:
        for msg in mid_iter:
            current_time += msg.time

            event_type = None
            if msg.type == "note_on":
                event_type = "pressed" if msg.velocity > 0 else "released"
            elif msg.type == "note_off":
                event_type = "released"

            print(f"Event: {msg}, type: {event_type}, time: {current_time}")
            if event_type:
                aligned_time = max(current_time + video_time_offset, 0)
                evt = PianoEvent(msg.note, msg.velocity, aligned_time, event_type)
                evt.frame_index = int(aligned_time * fps)
                evt.midi_time = round(current_time, 4)
                print(evt.frame_index, aligned_time, current_time, msg.type, msg)

                # Context Window: [-10, event, +10]
                start = max(evt.frame_index - 10, 0)
                end = min(evt.frame_index + 11, len(hands_data))
                evt.hand_context = hands_data[start:end]

                # Figure out which finger was used (if possible) by checking the current frame
                # and seeing which finger is closest to the piano key
                hands = hands_data[evt.frame_index]
                key_bb = key_bbs[evt.note]
                evt.key_bb = key_bb

                # Get dimensions for coordinate conversion
                # (We use mid_frame dims, assuming video res is constant)
                img_h, img_w = mid_frame.shape[:2]

                # Calculate center of the target key
                k_x1, k_y1, k_x2, k_y2 = key_bb
                key_center_x = (k_x1 + k_x2) / 2
                key_center_y = (k_y1 + k_y2) / 2

                min_dist = float("inf")
                best_finger = None
                best_hand = None

                for hand_i, hand in enumerate(hands):
                    # MediaPipe Fingertip Indices
                    for finger in FINGER_TIPS:
                        lm = hand["landmarks"][finger]
                        x, y = lm["x"], lm["y"]

                        # Convert normalized (0-1) to pixels
                        px, py = int(x * img_w), int(y * img_h)

                        dist = abs(px - key_center_x)

                        # Update best match
                        if dist < min_dist:
                            min_dist = dist
                            best_finger = finger
                            best_hand = hand_i

                # Threshold: If the closest finger is > 100px away,
                # it's likely a false positive or the hand wasn't detected.
                if min_dist < 100:
                    print(
                        f"Hand Event: Note {evt.note} {evt.event_type} by {best_hand} hand, finger {best_finger}"
                    )
                    evt.finger_used = best_finger
                    evt.hand_used = best_hand

                    if debug:
                        # Read the frame from the video and draw a line between the finger and key
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, evt.frame_index)
                        _, frame = cap.read()
                        cap.release()
                        print("event type", evt.event_type)
                        lm = hands[best_hand]["landmarks"][best_finger]
                        if lm:
                            img_h, img_w = frame.shape[:2]
                            px, py = int(lm["x"] * img_w), int(lm["y"] * img_h)
                            cv2.line(
                                frame,
                                (px, py),
                                (int(key_center_x), int(key_center_y)),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
                            cv2.circle(
                                frame,
                                (int(key_center_x), int(key_center_y)),
                                5,
                                (0, 0, 255),
                                -1,
                            )
                            # Draw entire hand at that frame
                            for hand in hands:
                                mp_draw.draw_landmarks(
                                    frame,
                                    PseudoLandmarkList(hand["landmarks"]),
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_draw.DrawingSpec(
                                        color=(0, 255, 255),
                                        thickness=2,
                                        circle_radius=2,
                                    ),
                                    mp_draw.DrawingSpec(
                                        color=(255, 0, 255), thickness=2
                                    ),
                                )
                            cv2.imshow("Finger to Key", frame)
                            while cv2.waitKey(10000000) != ord("q"):
                                pass

                else:
                    print(
                        f"Warning: Could not confidently determine finger/hand for event, distance of {min_dist}px too high.",
                        evt,
                    )
                    evt.finger_used = None
                    evt.hand_used = None

                events.append(evt)

            pbar.update(msg.time)

    # Sample non-event frames for balanced dataset
    # Create a unique set of (frame_index, hand_index, finger_index) tuples that have events
    unique_events = set([(e.frame_index, e.hand_used, e.finger_used) for e in events])
    print("Adding non-event frames...")
    n_key_presses = sum(1 for e in events if e.event_type == "pressed")
    sampled_non_events = 0
    while sampled_non_events < n_key_presses:
        # Randomly sample a frame, hand, and finger that does not have an event occur
        rand_frame = random.randint(21, len(hands_data) - 22)
        hands = hands_data[rand_frame]
        if not hands:
            continue

        rand_hand_i = random.randint(0, len(hands) - 1)
        hand = hands[rand_hand_i]
        rand_finger = random.choice(
            [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
        )
        if (rand_frame, rand_hand_i, rand_finger) in unique_events:
            continue

        evt = PianoEvent(
            note=None,
            velocity=0,
            timestamp=rand_frame / fps,
            event_type="no_event",
        )
        evt.frame_index = rand_frame
        start = max(evt.frame_index - 10, 0)
        end = min(evt.frame_index + 11, len(hands_data))
        evt.hand_context = hands_data[start:end]
        evt.hand_used = rand_hand_i
        evt.finger_used = rand_finger

        events.append(evt)
        sampled_non_events += 1
    print("Sampled", sampled_non_events, "non-event frames.")

    # 4. Write to JSON
    print(f"3. Writing {len(events)} events to {output_json_path}...")

    # Convert all Event objects to dicts
    json_data = {
        "metadata": {
            "total_frames": total_frames,
            "fps": fps,
            "video_path": video_path,
            "midi_path": midi_path,
            "piano_bb": tuple(int(coord) for coord in piano_bb),
            "key_bbs": key_bbs,
            "black_key_width": black_width,
            "black_key_height": black_height,
        },
        "events": [evt.to_dict() for evt in events],
    }

    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=None)  # indent=None keeps file size smaller

    print("Done!")


if __name__ == "__main__":
    piano_seg_model_path = "models/piano_seg.pt"
    midi_file = "data/coldplay-christmas-lights/midi.mid"
    video_file = "data/coldplay-christmas-lights/video.mp4"
    hand_output_file = "data/coldplay-christmas-lights/hand_data.json"
    output_file = "data/coldplay-christmas-lights/events.json"
    process_and_save(
        piano_seg_model_path,
        midi_file,
        video_file,
        hand_output_file,
        output_file,
        first_note_press_frame=58,
        black_width=15,
        black_height=100,
        debug=True,
    )

    with open(output_file, "r") as f:
        events = json.loads(f.read())["events"]
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            exit()

        for event in events:
            cap.set(cv2.CAP_PROP_POS_FRAMES, event["frame_index"])
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            for hand_i, hand in enumerate(event["hand_context"][5] or []):
                mp_draw.draw_landmarks(
                    frame,
                    PseudoLandmarkList(hand["landmarks"]),
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 255), thickness=2, circle_radius=2
                    ),
                    mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                )

                # Get wrist position for label placement
                h, w, _ = frame.shape
                wrist = hand["landmarks"][mp.solutions.hands.HandLandmark.WRIST]
                x, y = int(wrist["x"] * w), int(wrist["y"] * h)

                cv2.putText(
                    frame,
                    f"{hand['type']} Hand",
                    (x - 50, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Piano Hand Tracking", frame)

            # Press 'q' to quit, 'space' to pause
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                cv2.waitKey(0)  # Wait for another key press
