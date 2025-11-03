import cv2
import numpy as np

piano_img = cv2.imread("image.png")
width, height = piano_img.shape[1], piano_img.shape[0]

# --- 1. Define Dimensions and Colors ---

# 88-key piano has 52 white keys
NUM_WHITE_KEYS = 52

# Dimensions for individual keys (in pixels)
PADDING = [135, 55, 70, 35]  # top, right, bottom, left
WHITE_KEY_W = (width - PADDING[1] - PADDING[3]) / NUM_WHITE_KEYS
WHITE_KEY_H = height - PADDING[0] - PADDING[2]
BLACK_KEY_W = 30
BLACK_KEY_H = 130


# Image canvas dimensions
# Colors (OpenCV uses BGR format)
COLOR_BACKGROUND = (255, 255, 255)  # White
COLOR_BOX = (0, 0, 255)  # Red
BOX_THICKNESS = 2

# --- 2. Setup the Piano Key Pattern Logic ---

# In a C-Major scale (C,D,E,F,G,A,B), which white keys have a black key (a sharp)
# immediately after them?
# C (C#), D (D#), F (F#), G (G#), A (A#)
# E and B do not.

# Let's map C=0, D=1, E=2, F=3, G=4, A=5, B=6
# This set stores the indices of notes that *do* have a sharp.
notes_with_sharps = {0, 1, 3, 4, 5}

# A standard 88-key piano starts on the note 'A' (A0).
# In our C-based index, 'A' is index 5. This is our starting offset.
start_key_offset = 5

# --- 3. Create the Image and Draw Keys ---
print(f"Generating piano image ({width}x{height})...")

# Bounding box image
bb_img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)

# This will track the top-left x-coordinate of the *current white key*
current_x, current_y = PADDING[3], PADDING[0]


# Loop 52 times (once for each white key)
for i in range(NUM_WHITE_KEYS):

    # 1. Draw the white key bounding box
    # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    w_x1 = int(current_x)
    w_y1 = int(current_y)
    w_x2 = int(current_x + WHITE_KEY_W)
    w_y2 = int(current_y + WHITE_KEY_H)
    cv2.rectangle(bb_img, (w_x1, w_y1), (w_x2, w_y2), COLOR_BOX, BOX_THICKNESS)

    # Increment current_x to the start of the *next* white key
    current_x += WHITE_KEY_W

current_x, current_y = PADDING[3], PADDING[0]
# Loop 52 times (once for each white key)
for i in range(NUM_WHITE_KEYS):
    # 2. Determine the note index (0-6) for the *current* white key
    # (i + offset) % 7 gives the note name index (0=C, 1=D, ..., 6=B)
    note_index = (i + start_key_offset) % 7

    # 3. Conditionally draw the black key
    # We draw a black key if its note is in our `notes_with_sharps` set.
    # We also must check that we are not on the *very last* white key (i=51),
    # as there are no more keys after it.
    if note_index in notes_with_sharps and i < (NUM_WHITE_KEYS - 1):
        # The black key is centered on the *divider* between this
        # white key and the next one.
        b_center_x = current_x + WHITE_KEY_W
        b_x1 = b_center_x - (BLACK_KEY_W // 2)
        b_y1 = current_y
        b_x2 = b_center_x + (BLACK_KEY_W // 2)
        b_y2 = current_y + BLACK_KEY_H

        cv2.rectangle(
            bb_img, (int(b_x1), int(b_y1)), (int(b_x2), int(b_y2)), COLOR_BACKGROUND, -1
        )

        cv2.rectangle(
            bb_img,
            (int(b_x1), int(b_y1)),
            (int(b_x2), int(b_y2)),
            COLOR_BOX,
            BOX_THICKNESS,
        )

    # Increment current_x to the start of the *next* white key
    current_x += WHITE_KEY_W


print(f"Drawing complete. Found {NUM_WHITE_KEYS} white keys and 36 black keys.")

# --- 4. Display and Save the Image ---
cv2.imwrite("piano_bounding_boxes.png", bb_img)

# Overlay the generated bounding box image on top of the piano image
combined_img = cv2.addWeighted(piano_img, 0.7, bb_img, 0.3, 0)
cv2.imwrite("piano_comparison.png", combined_img)
