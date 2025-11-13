import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import make_directory_force_recursively
from dataset import dataset

train_transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),  # YOLO typically uses 640x640
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_loader, val_loader, test_loader = dataset(
    batch_size=32,
    split=(0.7, 0.15),
    transforms=(train_transform, val_transform, val_transform),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_wrapper = YOLO("yolov8n-cls.pt")  # YoloV8 nano
model = yolo_wrapper.model

# Replace the final layer for binary classification with dropout
for name, module in model.named_modules():
    if name == "model.9.linear":
        in_features = module.in_features
        # Replace with dropout + linear for binary classification
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        layer_name = name.rsplit(".", 1)[-1] if "." in name else name

        if parent_name:
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model

        setattr(
            parent,
            layer_name,
            nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 1)),
        )
        break

# Freeze all layers except the last layer
for name, param in model.named_parameters():
    if "model.9" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

num_epochs = 15

losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

make_directory_force_recursively("models/yolov8_piano_key")
torch.save(model.state_dict(), "models/yolov8_piano_key/model.pth")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), losses, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
make_directory_force_recursively("models/yolov8_piano_key/plots")
plt.savefig(
    "models/yolov8_piano_key/plots/training_loss.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Check accuracy, f1, etc. on validation and test sets
model.eval()


def evaluate(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


val_accuracy = evaluate(val_loader)
test_accuracy = evaluate(test_loader)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
