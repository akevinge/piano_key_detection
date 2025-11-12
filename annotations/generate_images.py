"""
If you created the annotation.json file manually, you can use this
script to save the entire frames to a directory for easier access.
"""

import cv2
import utils
import json

VIDEO_PATH = "videos/clown-balloon.mp4"
OUTPUT_DIR = "datasets/clown-balloon/raw"
ANNOTATION_FILE_PATH = "annotations/clown-balloon.mp4-annotations.json"

utils.make_directory_force_recursively(OUTPUT_DIR + "/images")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

annotations = json.loads(
    open(ANNOTATION_FILE_PATH).read()
)  # frame_i -> list of keys pressed

for frame_i in annotations.keys():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame:", frame_i)
        exit(1)

    file_name = f"{frame_i}.jpg"
    file_path = OUTPUT_DIR + "/images/" + file_name
    cv2.imwrite(file_path, frame)
    print(f"Saved frame {frame_i} to {file_path}")

# Copy annotations.json to OUTPUT_DIR
with open(OUTPUT_DIR + "/annotations.json", "w") as f:
    f.write(json.dumps(annotations, indent=4))
