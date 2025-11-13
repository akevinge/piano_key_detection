import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

import utils
from dataset import isolated_dataset_loader

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ds = isolated_dataset_loader(batch_size=32, transforms=val_transform)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 1))


for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

model.load_state_dict(
    torch.load("models/efficientnet_piano_key/model.pth", map_location=device)
)

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


val_accuracy = evaluate(ds)
print(f"Validation Accuracy: {val_accuracy:.4f}")
