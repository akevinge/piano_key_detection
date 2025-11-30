import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from dataset import dataset, isolated_dataset_loader
from utils import make_directory_force_recursively

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.efficientnet_b0(weights=None)
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


def collect_predictions(loader):
    """Collect all predictions and labels"""
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs.flatten())
            all_labels.extend(labels.flatten())

    return np.array(all_probs), np.array(all_labels)


def evaluate_at_threshold(probs, labels, threshold):
    """Calculate metrics at a specific threshold"""
    preds = (probs >= threshold).astype(int)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    # Accuracy
    accuracy = (preds == labels).mean()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_all_thresholds(probs, labels, thresholds):
    """Evaluate metrics across multiple thresholds"""
    results = {
        "thresholds": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for threshold in thresholds:
        metrics = evaluate_at_threshold(probs, labels, threshold)
        results["thresholds"].append(threshold)
        results["accuracy"].append(metrics["accuracy"])
        results["precision"].append(metrics["precision"])
        results["recall"].append(metrics["recall"])
        results["f1"].append(metrics["f1"])

    return results


def plot_metrics(results, dataset_name, save_path=None):
    """Plot precision, recall, and F1 vs threshold"""
    plt.figure(figsize=(10, 6))

    plt.plot(results["thresholds"], results["f1"], "r-", label="F1-Score", linewidth=2)
    plt.plot(
        results["thresholds"], results["accuracy"], "k--", label="Accuracy", linewidth=2
    )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(f"Metrics vs Threshold - {dataset_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Main evaluation
if __name__ == "__main__":

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # small shifts
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # train_loader, val_loader, test_loader = dataset(
    #     batch_size=32,
    #     split=(0.7, 0.15),
    #     transforms=(train_transform, val_transform, val_transform),
    # )

    # Define thresholds to evaluate
    thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00

    # Evaluate isolated dataset
    print("Evaluating isolated dataset...")
    isolated = isolated_dataset_loader(batch_size=32, transforms=transform)

    print("Collecting predictions...")
    isolated_probs, isolated_labels = collect_predictions(isolated)
    print(f"Total samples: {len(isolated_labels)}")

    print("\nEvaluating at different thresholds...")
    isolated_results = evaluate_all_thresholds(
        isolated_probs, isolated_labels, thresholds
    )

    # Find best threshold based on F1
    best_idx = np.argmax(isolated_results["f1"])
    best_threshold = isolated_results["thresholds"][best_idx]

    print(f"\nBest threshold (by F1): {best_threshold:.3f}")
    print(f"  Accuracy: {isolated_results['accuracy'][best_idx]:.4f}")
    print(f"  Precision: {isolated_results['precision'][best_idx]:.4f}")
    print(f"  Recall: {isolated_results['recall'][best_idx]:.4f}")
    print(f"  F1: {isolated_results['f1'][best_idx]:.4f}")

    # Plot results
    plot_metrics(
        isolated_results, "Isolated Dataset", "isolated_metrics_vs_threshold.png"
    )

    print("\nResults saved:")
    print("  - Plot: isolated_metrics_vs_threshold.png")
    print("  - MATLAB data: isolated_metrics.mat")
