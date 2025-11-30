import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
import numpy as np

# Piano settings
# 88-key piano has 52 white keys
NUM_WHITE_KEYS = 52
NUM_BLACK_KEYS = 36

# Let's map C=0, D=1, E=2, F=3, G=4, A=5, B=6
# This set stores the indices of notes that *do* have a sharp.
NOTES_WITH_SHARPS = {0, 1, 3, 4, 5}

# A standard 88-key piano starts on the note 'A' (A0).
# In our C-based index, 'A' is index 5. This is our starting offset.
START_KEY_OFFSET = 5


VIDEO_SOURCE = "videos/elegy.mp4"
PADDING = (455, 0, 415, 0)
BLACK_KEY_WIDTH = 25
BLACK_KEY_HEIGHT = 145

# VIDEO_SOURCE = "videos/bach-preludes.mp4"
# PADDING = (660, 40, 200, 34)
# BLACK_KEY_WIDTH = 25
# BLACK_KEY_HEIGHT = 145


STARTING_FRAME = 300


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


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.efficientnet_b0(weights=None)  # Don't load pretrained weights
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 1))

# Load trained weights
model.load_state_dict(
    torch.load("models/efficientnet_piano_key/model.pth", map_location=device)
)
model = model.to(device)
model.eval()

# Define the same transform used for validation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_i = STARTING_FRAME
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    key_inferences = {}
    # Perform inference on every key
    for key, key_bb in bounding_boxes.items():
        x1, y1, x2, y2 = key_bb
        # Get 25 pixels to left and right for all keys
        x1, x2 = max(0, x1 - 30), min(frame.shape[1], x2 + 30)
        y2 = min(frame.shape[0], y2 + 100)

        key_img = frame[y1:y2, x1:x2]

        # Convert to PIL Image
        pil_image = Image.fromarray(key_img)

        # Apply transforms
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = int(probability > 0.5)

        # Display results on frame
        label = f"Key: {key} {prediction} ({probability:.2%})"
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

        center_x, center_y = (y2 - y1) // 2, (x2 - x1) // 2
        key_inferences[key] = (prediction, probability)

        # Show the frame
    frame_i += 10

    for key, (is_pressed, prob) in key_inferences.items():
        x1, y1, x2, y2 = bounding_boxes[key]
        # Determine color and thickness based on key press state
        if is_pressed and prob > 0.65:
            color = (0, 255, 0)  # Green when pressed
            thickness = 3
        else:
            color = (255, 255, 255)  # White when not pressed
            thickness = 2

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("Piano Key Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
