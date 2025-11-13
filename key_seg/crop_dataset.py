import cv2
import numpy as np
import glob
import os
import json

import utils
from coco import CocoBuilder

# Video path
VIDEO_PATH = "videos/elegy.mp4"
RAW_IMAGE_PATH = "datasets/elegy/raw/images"
ANNOTATION_FILE_PATH = "datasets/elegy/raw/annotations.json"
OUTPUT_DIR = "datasets/elegy/cropped"


NON_PRESSED_NEIGHBOR_BIAS = 0.5

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Piano settings
# 88-key piano has 52 white keys
NUM_WHITE_KEYS = 52
NUM_BLACK_KEYS = 36

# Fur Elise
# PADDING = (672, 50, 220, 45)
# BLACK_KEY_WIDTH = 50
# BLACK_KEY_HEIGHT = 106

# Clown Balloon
# PADDING = (445, 35, 420, 40)
# BLACK_KEY_WIDTH = 25
# BLACK_KEY_HEIGHT = 135

# Chopin Etude Op.10
# PADDING = (410, 0, 440, 0)
# BLACK_KEY_WIDTH = 25
# BLACK_KEY_HEIGHT = 145

# Elegy
PADDING = (455, 0, 415, 0)
BLACK_KEY_WIDTH = 25
BLACK_KEY_HEIGHT = 145

# Top, right, bottom, left padding for the piano in the video
# Increase top to move the piano down, increase bottom to move it up.
# Increase both will shrink the piano vertically.
# The same applies for left/right.

# In a C-Major scale (C,D,E,F,G,A,B), which white keys have a black key (a sharp)
# immediately after them?
# C (C#), D (D#), F (F#), G (G#), A (A#)
# E and B do not.

# Let's map C=0, D=1, E=2, F=3, G=4, A=5, B=6
# This set stores the indices of notes that *do* have a sharp.
NOTES_WITH_SHARPS = {0, 1, 3, 4, 5}

# A standard 88-key piano starts on the note 'A' (A0).
# In our C-based index, 'A' is index 5. This is our starting offset.
START_KEY_OFFSET = 5


def get_key_neighbors(key: str) -> list[str]:
    if key.lower().startswith("b"):
        key_i = int(key[1:])
        neighbors = []
        if key_i > 0:
            neighbors.append("b" + str(key_i - 1))
        if key_i < NUM_BLACK_KEYS - 1:
            neighbors.append("b" + str(key_i + 1))
        return neighbors
    else:
        key_i = int(key)
        neighbors = []
        if key_i > 0:
            neighbors.append(str(key_i - 1))
        if key_i < NUM_WHITE_KEYS - 1:
            neighbors.append(str(key_i + 1))
        return neighbors


def generate_piano_bbs(
    width: int,
    height: int,
    padding: tuple[int, int, int, int],
    black_width: int,
    black_height: int,
) -> dict[str, tuple[int, int, int, int]]:
    bounding_boxes = {}
    top, right, bottom, left = padding

    # Dimensions for individual keys (in pixels)
    WHITE_KEY_W = (width - left - right) / NUM_WHITE_KEYS
    WHITE_KEY_H = height - top - bottom

    # This will track the top-left x-coordinate of the *current white key*
    current_x, current_y = left, top

    # Loop 52 times (once for each white key)
    for i in range(NUM_WHITE_KEYS):

        # 1. Draw the white key bounding box
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        w_x1 = int(current_x)
        w_y1 = int(current_y)
        w_x2 = int(current_x + WHITE_KEY_W)
        w_y2 = int(current_y + WHITE_KEY_H)

        bounding_boxes[str(i)] = (w_x1, w_y1, w_x2, w_y2)

        # Increment current_x to the start of the *next* white key
        current_x += WHITE_KEY_W

    current_x, current_y = left, top
    black_key_i = 0
    # Loop 52 times (once for each white key)
    for i in range(NUM_WHITE_KEYS):
        # 2. Determine the note index (0-6) for the *current* white key
        # (i + offset) % 7 gives the note name index (0=C, 1=D, ..., 6=B)
        note_index = (i + START_KEY_OFFSET) % 7

        # 3. Conditionally draw the black key
        # We draw a black key if its note is in our `notes_with_sharps` set.
        # We also must check that we are not on the *very last* white key (i=51),
        # as there are no more keys after it.
        if note_index in NOTES_WITH_SHARPS and i < (NUM_WHITE_KEYS - 1):
            # The black key is centered on the *divider* between this
            # white key and the next one.
            b_center_x = current_x + WHITE_KEY_W
            b_x1 = b_center_x - (black_width // 2)
            b_y1 = current_y
            b_x2 = b_center_x + (black_width // 2)
            b_y2 = current_y + black_height

            bounding_boxes[f"b{black_key_i}"] = (
                int(b_x1),
                int(b_y1),
                int(b_x2),
                int(b_y2),
            )

            black_key_i += 1

        # Increment current_x to the start of the *next* white key
        current_x += WHITE_KEY_W

    return bounding_boxes


bounding_boxes = generate_piano_bbs(
    width=1920,
    height=1080,
    padding=PADDING,
    black_width=BLACK_KEY_WIDTH,
    black_height=BLACK_KEY_HEIGHT,
)
print(bounding_boxes)

# Loop through the raw images
image_files = []
for filepath in glob.glob(f"{RAW_IMAGE_PATH}/*"):
    if os.path.isfile(filepath) and os.path.exists(filepath):
        image_files.append(filepath)

# Load annoattions file
annotations = json.loads(open(ANNOTATION_FILE_PATH, "r").read())

dataset = CocoBuilder([{"id": 0, "name": "not_pressed"}, {"id": 1, "name": "pressed"}])

utils.force_remove_directory(OUTPUT_DIR)
utils.make_directory_force_recursively(OUTPUT_DIR)
for img_path in image_files:
    frame_i = os.path.basename(img_path)[:-4]
    keys_pressed = annotations[frame_i]

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame:", frame_i)
        exit(1)

    try:
        for key in keys_pressed:
            # Crop out the bounding box for this key
            x1, y1, x2, y2 = bounding_boxes[str(key)]
            # Get 25 pixels to left and right for all keys
            x1, x2 = max(0, x1 - 30), min(frame.shape[1], x2 + 30)
            y2 = min(frame.shape[0], y2 + 100)

            key_img = frame[y1:y2, x1:x2]

            file_name = f"{frame_i}-{key}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), key_img)

            image_id = dataset.add_image(
                file_name=file_name, width=key_img.shape[1], height=key_img.shape[0]
            )
            bbox = [0, 0, x2 - x1, y2 - y1]
            category_id = 1
            dataset.add_annotation(
                image_id=image_id, bbox=bbox, category_id=category_id
            )

        non_pressed_keys = [
            k for k in range(NUM_WHITE_KEYS) if str(k) not in keys_pressed
        ] + [f"b{k}" for k in range(NUM_BLACK_KEYS) if f"b{k}" not in keys_pressed]

        # Take a random sample of non-pressed keys, biased towards pressed-key neighbors
        neighboring_keys = set()
        for k in keys_pressed:
            neighbors = get_key_neighbors(k)
            for n in neighbors:
                neighboring_keys.add(n)

        sample_size = min(len(keys_pressed), len(non_pressed_keys))
        sampled_non_pressed = []
        for _ in range(sample_size):
            if neighboring_keys and np.random.rand() < NON_PRESSED_NEIGHBOR_BIAS:
                sampled_non_pressed.append(np.random.choice(list(neighboring_keys)))
            else:
                sampled_non_pressed.append(np.random.choice(non_pressed_keys))

        for key in sampled_non_pressed:
            x1, y1, x2, y2 = bounding_boxes[str(key)]
            # Get 25 pixels to left and right for all keys
            x1, x2 = max(0, x1 - 30), min(frame.shape[1], x2 + 30)
            y1 = max(0, y1 - 20)
            y2 = min(frame.shape[0], y2 + 100)

            key_img = frame[y1:y2, x1:x2]

            file_name = f"{frame_i}-{key}-np.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), key_img)
            image_id = dataset.add_image(
                file_name=file_name, width=key_img.shape[1], height=key_img.shape[0]
            )
            bbox = [0, 0, x2 - x1, y2 - y1]
            category_id = 0
            dataset.add_annotation(
                image_id=image_id, bbox=bbox, category_id=category_id
            )
    except Exception as e:
        print("Failed to process image:", img_path, "Error:", e)

dataset.save(f"{OUTPUT_DIR}/annotations.json")
