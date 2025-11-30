from ultralytics import YOLO


class PianoSegmentationModel:
    def __init__(self, model_path="piano_seg.pt"):
        self.model = YOLO(model_path, task="segment")

    def segment_piano(self, image, save=False, show=False) -> tuple[int, int, int, int]:
        results = self.model.predict(source=image, save=save, show=show)
        x1, y1, x2, y2 = results[0].boxes.xyxy[0].numpy().astype(int)
        return x1, y1, x2, y2


def generate_piano_bbs_88_key(
    width: int,
    height: int,
    padding: tuple[int, int, int, int],
    black_width: int,
    black_height: int,
) -> dict[int, tuple[int, int, int, int]]:  # Returns Dict[MIDI_NOTE, Box]

    NUM_WHITE_KEYS = 52

    # --- STEP 1: Generate the MIDI Map ---
    # We map your visual indices (white 0-51, black 0-35) to MIDI notes (21-108)
    white_key_midi_map = {}
    black_key_midi_map = {}

    current_midi = 21  # Start at A0 (MIDI 21)
    black_key_count = 0

    # We iterate through the white keys logically to assign MIDI numbers
    for i in range(NUM_WHITE_KEYS):
        # 1. Assign current MIDI to the current White Key
        white_key_midi_map[i] = current_midi
        current_midi += 1

        # 2. Check if there is a black key to the right
        # A=0, B=1, C=2, D=3, E=4, F=5, G=6 (This is a 0-6 index relative to A)
        # We know A, C, D, F, G have sharps.
        # Relative to A (index 0): A(0), C(2), D(3), F(5), G(6) have sharps
        note_index_from_a = i % 7
        has_sharp = note_index_from_a in {0, 2, 3, 5, 6}

        if has_sharp and i < (NUM_WHITE_KEYS - 1):
            black_key_midi_map[black_key_count] = current_midi
            current_midi += 1
            black_key_count += 1

    # --- STEP 2: Draw the Boxes (Using the Map) ---
    # Now we run your original drawing logic, but use the maps above for keys
    NOTES_WITH_SHARPS = {0, 1, 3, 4, 5}
    START_KEY_OFFSET = 5  # Start at A

    bounding_boxes = {}
    top, right, bottom, left = padding
    WHITE_KEY_W = (width - left - right) / NUM_WHITE_KEYS
    WHITE_KEY_H = height - top - bottom

    current_x, current_y = left, top

    # Loop 1: White Keys
    for i in range(NUM_WHITE_KEYS):
        w_x1, w_y1 = int(current_x), int(current_y)
        w_x2, w_y2 = int(current_x + WHITE_KEY_W), int(current_y + WHITE_KEY_H)

        # KEY CHANGE: Use the MIDI note as the dictionary key
        midi_note = white_key_midi_map[i]
        bounding_boxes[midi_note] = (w_x1, w_y1, w_x2, w_y2)

        current_x += WHITE_KEY_W

    # Loop 2: Black Keys
    current_x = left
    black_key_i = 0

    for i in range(NUM_WHITE_KEYS):
        note_index = (i + START_KEY_OFFSET) % 7  # C-relative index

        if note_index in NOTES_WITH_SHARPS and i < (NUM_WHITE_KEYS - 1):
            b_center_x = current_x + WHITE_KEY_W
            b_x1 = int(b_center_x - (black_width // 2))
            b_y1 = int(current_y)
            b_x2 = int(b_center_x + (black_width // 2))
            b_y2 = int(current_y + black_height)

            # KEY CHANGE: Use the MIDI note as the dictionary key
            midi_note = black_key_midi_map[black_key_i]
            bounding_boxes[midi_note] = (b_x1, b_y1, b_x2, b_y2)

            black_key_i += 1

        current_x += WHITE_KEY_W

    return bounding_boxes
