import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import json
import copy

import mediapipe as mp


from midi import PianoEvent

mp_hands = mp.solutions.hands

CONTEXT_LENGTH = 21
LABEL_MAP = {"no_event": 0, "pressed": 1, "released": 2}


def load_metadata(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["metadata"]


def load_data(data_paths: list[str]) -> list[PianoEvent]:
    data = []
    for path in data_paths:
        with open(path) as f:
            raw_data = json.load(f)
            events = [PianoEvent(**evt) for evt in raw_data["events"]]
            # For each landmark in each hand in each event, truncate to 11 elements
            for event in events:
                event.hand_context = event.hand_context[:CONTEXT_LENGTH]

            data.extend(events)

    return data


def euclidean_distance(lm1, lm2):
    return np.sqrt((lm1["x"] - lm2["x"]) ** 2 + (lm1["y"] - lm2["y"]) ** 2)


def z_score_normalize(array: np.ndarray):
    """Applies Z-score normalization to the input array."""
    return (array - np.mean(array)) / (np.std(array) + 1e-6)


def min_max_normalize(array: np.ndarray):
    """Applies Min-Max normalization to the input array."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return array - min_val  # Avoid division by zero
    return (array - min_val) / (max_val - min_val)


def apply_standardization(array: np.ndarray):
    """Apply to all features for consistency"""
    return z_score_normalize(array)


def feat_palm_scale(event: PianoEvent):
    """
    Calculates the 'size' of the hand (Wrist to Middle Finger MCP).
    Normalizes relative to a reasonable starting frame in the window to detect hand 'drops'.
    """
    # Indices: Wrist=0, Middle_MCP=9
    scales = []

    # Hand contex includes 10 frames before and after (CONTEXT_LENGTH frames total).
    # Let's use a reasonable starting point before the motion for scale.
    base_frame = 5
    used_hand_landmarks = event.get_used_hand_landmarks(offset=base_frame)
    wrist = used_hand_landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = used_hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    dist = euclidean_distance(wrist, middle_mcp)
    base_scale = dist if dist > 0 else 1.0

    for i in range(len(event.hand_context)):
        lm_list = event.get_used_hand_landmarks(offset=i)
        wrist = lm_list[mp_hands.HandLandmark.WRIST]
        middle_mcp = lm_list[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # Euclidean distance
        dist = euclidean_distance(wrist, middle_mcp)

        scales.append(dist / base_scale)

    return apply_standardization(np.array(scales))


def feat_finger_curvature(event: PianoEvent):
    """
    Captures finger curling motion.
    """
    dists = []
    for i in range(len(event.hand_context)):
        landmarks = event.get_used_hand_landmarks(offset=i)
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        tip = event.get_used_fingertip_landmark(offset=i)

        dists.append(euclidean_distance(wrist, tip))

    dists = np.array(dists)
    return dists


def feat_tip_velocity(event: PianoEvent):
    """
    Calculates instantaneous velocity of the tip.
    """
    vels = [0.0]  # Start with 0 velocity for the first frame.
    for i in range(1, len(event.hand_context)):
        curr_tip, prev_tip = (
            event.get_used_fingertip_landmark(offset=i),
            event.get_used_fingertip_landmark(offset=i - 1),
        )

        vels.append(euclidean_distance(curr_tip, prev_tip))

    return np.array(vels)


def feat_neighbor_contrast(event: PianoEvent):
    """
    Calculates the length difference between the active finger and its neighbors.
    If Index (8) is active, compare with Middle (12).
    """
    if event.finger_used is None:
        return None

    # Map fingers to their comparison neighbor (Tip Indices)
    # Thumb(4)->Index(8), Index(8)->Middle(12), Middle(12)->Ring(16), Ring(16)->Pinky(20), Pinky(20)->Ring(16)
    neighbor_map = {
        mp_hands.HandLandmark.THUMB_TIP: mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP: mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP: mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP: mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_TIP: mp_hands.HandLandmark.RING_FINGER_TIP,
    }

    neighbor_tip_idx = neighbor_map.get(
        event.finger_used, mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    )

    diffs = []

    for i in range(len(event.hand_context)):
        landmarks = event.get_used_hand_landmarks(offset=i)
        wrist = landmarks[mp_hands.HandLandmark.WRIST]

        # Calculate Active Finger Length (Wrist to Tip)
        active_tip = landmarks[event.finger_used]
        len_active = euclidean_distance(wrist, active_tip)

        # Calculate Neighbor Finger Length
        neighbor_tip = landmarks[neighbor_tip_idx]
        len_neighbor = euclidean_distance(wrist, neighbor_tip)

        # The Feature: The difference.
        # If Active presses, len_active shrinks, len_neighbor stays same -> distinct signal.
        diffs.append(len_active - len_neighbor)

    return apply_standardization(np.array(diffs))


def feat_tip_acceleration(event: PianoEvent):
    """
    Calculates the change in velocity (Acceleration).
    """
    vels = feat_tip_velocity(event)

    # 2. Calculate diff (Acceleration)
    # np.diff reduces size by 1, so we pad the start
    accels = np.diff(vels, prepend=vels[0])

    return accels


def feat_vertical_position(event: PianoEvent):
    """Y-coordinate of fingertip (key to detecting downward press)"""
    positions = []
    for i in range(len(event.hand_context)):
        tip = event.get_used_fingertip_landmark(offset=i)
        positions.append(tip["y"])  # Y increases downward in image coords
    return apply_standardization(np.array(positions))


def feat_tip_to_key_distance(event: PianoEvent):
    """If you have key positions, distance to target key"""
    distances = []
    key_box = event.key_bb
    if key_box is None:
        mid_frame = len(event.hand_context) // 2
        mid_tip = event.get_used_fingertip_landmark(offset=mid_frame)
        if mid_tip is None:
            return None
        key_center_x = mid_tip["x"]
    else:
        key_center_x = (key_box[0] + key_box[2]) / 2

    for i in range(len(event.hand_context)):
        tip = event.get_used_fingertip_landmark(offset=i)
        if tip is None:
            return None
        distances.append(abs(tip["x"] - key_center_x))

    return apply_standardization(np.array(distances))


def feat_curvature_velocity(event: PianoEvent):
    """How fast is the finger curling/uncurling"""
    curvatures = feat_finger_curvature(event)
    curv_vel = np.diff(curvatures, prepend=curvatures[0])
    return curv_vel


def feat_motion_direction_ratio(event: PianoEvent):
    """
    Ratio of frames with downward vs upward motion in the window.
    Press events should have more downward motion, release more upward.
    """
    ratios = []
    window_size = 5  # Look at 5 frames at a time

    for center in range(len(event.hand_context)):
        start = max(0, center - window_size // 2)
        end = min(len(event.hand_context), center + window_size // 2 + 1)

        down_count = 0
        up_count = 0

        for i in range(start + 1, end):
            curr_tip = event.get_used_fingertip_landmark(offset=i)
            prev_tip = event.get_used_fingertip_landmark(offset=i - 1)

            y_diff = curr_tip["y"] - prev_tip["y"]
            if y_diff > 0.001:  # Small threshold to ignore noise
                down_count += 1
            elif y_diff < -0.001:
                up_count += 1

        total = down_count + up_count
        if total > 0:
            ratio = (
                down_count - up_count
            ) / total  # Range: -1 (all up) to 1 (all down)
        else:
            ratio = 0
        ratios.append(ratio)

    return np.array(ratios)


# --- CONFIGURATION ---
# To add/remove features, just edit this list.
# The model will automatically resize itself.
ACTIVE_FEATURES = [
    # lambda event: feat_palm_scale(event),
    lambda event: feat_finger_curvature(event),
    lambda event: feat_tip_velocity(event),
    # lambda event: feat_neighbor_contrast(event),
    lambda event: feat_tip_acceleration(event),
    # lambda event: feat_vertical_position(event),
    lambda event: feat_tip_to_key_distance(event),
    lambda event: feat_curvature_velocity(event),
    lambda event: feat_motion_direction_ratio(event),
]


def collate_skip_erroneous(batch):
    """
    Filters out None values from the batch.
    """
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])  # Or handle specifically

    return default_collate(batch)


def extract_features(event: PianoEvent, feature_extractors, augment=False):
    feature_channels = []

    try:
        for extractor in feature_extractors:
            feat_array = extractor(event)

            # Check length requirements
            if len(feat_array) != CONTEXT_LENGTH:
                # If you want to be strict and skip bad lengths:
                return None

            feature_channels.append(feat_array)

        # Stack features
        input_tensor = torch.tensor(np.stack(feature_channels), dtype=torch.float32)

        # Get Label
        label_value = LABEL_MAP.get(event.event_type, 0)
        label_tensor = torch.tensor(label_value, dtype=torch.long)

        return input_tensor, label_tensor
    except Exception as e:
        # print(f"Skipping bad sample: {e}")
        return None


class PianoPressDataset(Dataset):
    def __init__(self, events: list[PianoEvent], feature_extractors, augment=False):
        """
        events_json_data: The list of dicts loaded from your JSON file.
        feature_extractors: List of functions to run on the frames.
        """
        self.data: list[PianoEvent] = [
            ev for ev in events if ev.has_finger_key_matched()
        ]
        self.extractors = feature_extractors
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        event = self.data[idx]
        return extract_features(event, self.extractors, augment=self.augment)


class PianoPressNet(nn.Module):
    def __init__(self, num_features, seq_len=CONTEXT_LENGTH, num_classes=3):
        super(PianoPressNet, self).__init__()

        self.seq_len = seq_len

        # 1. Feature Extractor (Same as before)
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Conv1d(
                in_channels=num_features, out_channels=16, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Calculate Flatten Dim
        dummy_input = torch.zeros(1, num_features, seq_len)
        with torch.no_grad():
            dummy_out = self.feature_extractor(dummy_input)
            self.flatten_dim = dummy_out.shape[1] * dummy_out.shape[2]

        # 2. Classification Head (CHANGED)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output size is now num_classes (3)
            nn.Linear(64, num_classes),
            # NOTE: No Sigmoid/Softmax here!
            # CrossEntropyLoss expects raw "logits".
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    train_paths = [
        "data/the-pogues/events.json",
    ]

    train_data = load_data(train_paths)
    train_dataset = PianoPressDataset(train_data, ACTIVE_FEATURES, augment=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_skip_erroneous
    )

    eval_paths = [
        "data/have-yourself-a-merry-little-christmas/events.json",
    ]
    eval_data = load_data(eval_paths)
    eval_dataset = PianoPressDataset(eval_data, ACTIVE_FEATURES)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=32, shuffle=True, collate_fn=collate_skip_erroneous
    )

    # Check class distribution
    label_counts = [0, 0, 0]
    for i in range(len(train_dataset)):
        train_data = train_dataset[i]
        if train_data is not None:
            _, label = train_data
            label_counts[label.item()] += 1

    print(
        f"Class distribution: No Event: {label_counts[0]}, Pressed: {label_counts[1]}, Released: {label_counts[2]}"
    )

    print(f"Training on {sum(label_counts)} samples.")

    model = PianoPressNet(
        num_features=len(ACTIVE_FEATURES), seq_len=CONTEXT_LENGTH, num_classes=3
    )

    # 3. Loss & Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # CHANGED: Use CrossEntropyLoss for multi-class
    class_weights = torch.tensor([0.5, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 4. Loop

    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            # Outputs shape: (Batch, 3)
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate Accuracy (Optional but helpful)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch+1} | Loss: {total_loss/len(train_dataloader):.4f} | Acc: {100 * correct / total:.2f}%"
        )

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print("Class-wise Accuracy:")
    for i in range(3):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  Class {i}: No samples.")

    # Save model
    torch.save(model.state_dict(), "models/handtrack_piano_press_net.pth")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score

    # --- 1. Data Collection ---
    model.eval()
    all_probs = []
    all_targets = []

    print("Collecting model predictions...")
    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs)  # Shape: (N_samples, N_classes)
    all_targets = np.concatenate(all_targets)  # Shape: (N_samples,)
    n_classes = all_probs.shape[1]

    # --- 2. Threshold Sweeping ---
    thresholds = np.arange(0.05, 1.0, 0.05)
    results = {i: {"thresh": [], "p": [], "r": [], "f1": []} for i in range(n_classes)}

    print("Calculating metrics...")
    for cls_idx in range(n_classes):
        # Binary target for the current class (One-vs-Rest)
        y_true_binary = (all_targets == cls_idx).astype(int)

        for t in thresholds:
            # If prob > t, predict 1, else 0
            preds_binary = (all_probs[:, cls_idx] >= t).astype(int)

            # Calculate metrics
            p = precision_score(y_true_binary, preds_binary, zero_division=0)
            r = recall_score(y_true_binary, preds_binary, zero_division=0)
            f = f1_score(y_true_binary, preds_binary, zero_division=0)

            results[cls_idx]["thresh"].append(t)
            results[cls_idx]["p"].append(p)
            results[cls_idx]["r"].append(r)
            results[cls_idx]["f1"].append(f)

    # --- 3. Plotting ---
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)
    if n_classes == 1:
        axes = [axes]  # Handle single class case

    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        data = results[cls_idx]

        # Plot curves
        ax.plot(data["thresh"], data["p"], "--", label="Precision", alpha=0.7)
        ax.plot(data["thresh"], data["r"], "--", label="Recall", alpha=0.7)
        ax.plot(
            data["thresh"],
            data["f1"],
            "-",
            label="F1 Score",
            linewidth=2,
            color="green",
        )

        # Find and annotate best F1
        best_idx = np.argmax(data["f1"])
        best_val = data["f1"][best_idx]
        best_thresh = data["thresh"][best_idx]

        ax.scatter(best_thresh, best_val, color="red", zorder=5)
        ax.annotate(
            f"Best F1: {best_val:.2f}\n@ {best_thresh:.2f}",
            (best_thresh, best_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

        ax.set_title(f"Class {cls_idx}")
        ax.set_xlabel("Threshold")
        ax.grid(True, alpha=0.3)
        if cls_idx == 0:
            ax.set_ylabel("Score")
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
