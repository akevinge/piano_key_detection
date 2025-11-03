import cv2
import os
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import *
import json
from PIL import Image, ImageTk

INPUT_VIDEO_PATH = "fur-elise.mp4"
ANNOTATIONS_FOLDER = "annotations/"
FRAME_SKIP = (
    2  # How many frames to skip between annotations. 0 to annotate every frame.
)

# Piano settings
# 88-key piano has 52 white keys
NUM_WHITE_KEYS = 52
NUM_BLACK_KEYS = 36

# Top, right, bottom, left padding for the piano in the video
# Increase top to move the piano down, increase bottom to move it up.
# Increase both will shrink the piano vertically.
# The same applies for left/right.
PADDING = (672, 50, 220, 45)
BLACK_KEY_WIDTH = 50
BLACK_KEY_HEIGHT = 106

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


# --- 1. Define Dimensions and Colors ---
def generate_piano_bb_image(
    width: int,
    height: int,
    padding: tuple[int, int, int, int],
    black_width: int,
    black_height: int,
) -> npt.NDArray:
    top, right, bottom, left = padding

    # Dimensions for individual keys (in pixels)
    WHITE_KEY_W = (width - left - right) / NUM_WHITE_KEYS
    WHITE_KEY_H = height - top - bottom

    # Image canvas dimensions
    # Colors (OpenCV uses BGR format)
    COLOR_BACKGROUND = (255, 255, 255)  # White
    COLOR_BOX = (0, 0, 255)  # Red
    BOX_THICKNESS = 2

    # --- 2. Setup the Piano Key Pattern Logic ---

    # --- 3. Create the Image and Draw Keys ---
    print(f"Generating piano image ({width}x{height})...")

    # Bounding box image
    bb_img = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)

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
        cv2.rectangle(bb_img, (w_x1, w_y1), (w_x2, w_y2), COLOR_BOX, BOX_THICKNESS)
        # Add the number label (0-51) at the bottom of the white key
        cv2.putText(
            img=bb_img,
            text=str(i),
            org=(w_x1 + 5, w_y2 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=COLOR_BOX,
            thickness=0,
            lineType=cv2.LINE_AA,
        )

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

            # Erase the area with background color first to ensure that
            # black key is drawn on top of white keys.
            cv2.rectangle(
                bb_img,
                (int(b_x1), int(b_y1)),
                (int(b_x2), int(b_y2)),
                COLOR_BACKGROUND,
                -1,
            )

            cv2.rectangle(
                bb_img,
                (int(b_x1), int(b_y1)),
                (int(b_x2), int(b_y2)),
                COLOR_BOX,
                BOX_THICKNESS,
            )

            cv2.putText(
                img=bb_img,
                text=f"{black_key_i}",
                org=(int(b_x1) + black_width // 2 - 5, int(b_y1) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                # White text on black key
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            black_key_i += 1
        # Increment current_x to the start of the *next* white key
        current_x += WHITE_KEY_W

    return bb_img


def prompt_for_keys():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    user_input = simpledialog.askstring(
        "Annotate Frame", "Enter pressed keys (e.g. 0-51, b0-b49):"
    )
    root.destroy()
    if not user_input:
        return set()
    return {k.strip().upper() for k in user_input.split(",") if k.strip()}


cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

root = Tk()
root.title("Piano Key Annotation")

# --- UI Components ---
frame_label = Label(root)
frame_label.pack()

entry_label = Label(root, text="Enter keys pressed (e.g. C4,E4,G4):")
entry_label.pack()

entry = Entry(root, width=50)
entry.pack()

annotation_label = Label(root, text="", fg="blue")
annotation_label.pack()

frame_idx = 0
annotations = {}

# Check if file exists:
if os.path.exists(f"{INPUT_VIDEO_PATH}-annotations.json"):
    print("Are you sure want to overwrite existing annotations? (y/n): ", end="")
    choice = input().strip().lower()
    if choice != "y":
        print("Exiting without overwriting.")
        exit()


os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
file = open(f"{ANNOTATIONS_FOLDER}/{INPUT_VIDEO_PATH}-annotations.json", "w")


def validate_entry():
    keys = entry.get().strip()
    if keys == "":
        return True
    keys = [k.strip().lower() for k in keys.split(",")]
    try:
        for k in keys:
            if k.startswith("b"):
                k_int = int(k[1:])
                if k_int not in range(NUM_BLACK_KEYS):
                    return False
            elif k.isdigit():
                k_int = int(k)
                if k_int not in range(NUM_WHITE_KEYS):
                    return False
            else:
                # K is not a number or starts with b
                return False
    except:
        return False

    return True


# --- Display & Annotation Logic ---
def show_next_frame():
    global frame_idx
    keys = entry.get().strip()
    if not validate_entry():
        annotation_label.config(
            text="Invalid input. Use comma-separated numbers or 'b'-prefixed black keys (e.g., 1, b3, 7)."
        )
        return  # don't advance frames, let user retry

    if keys:
        annotations[frame_idx - FRAME_SKIP - 1] = keys
    file.seek(0)
    file.write(json.dumps(annotations, indent=4))
    file.flush()

    ret, frame = cap.read()
    if not ret:
        annotation_label.config(text="End of video reached.")
        return

    bb_img = generate_piano_bb_image(
        frame.shape[1],
        frame.shape[0],
        padding=PADDING,
        black_width=BLACK_KEY_WIDTH,
        black_height=BLACK_KEY_HEIGHT,
    )
    combined_img = cv2.addWeighted(frame, 0.7, bb_img, 0.3, 0)
    display_width = 960
    display_height = int(
        combined_img.shape[0] * (display_width / combined_img.shape[1])
    )
    resized = cv2.resize(combined_img, (display_width, display_height))

    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)
    entry.delete(0, END)

    frame_idx += FRAME_SKIP + 1
    root.after(10, lambda: None)  # keep the UI responsive


Button(root, text="Next Frame", command=show_next_frame).pack()
Button(root, text="Quit", command=lambda: (cap.release(), root.destroy())).pack()

show_next_frame()
root.mainloop()
