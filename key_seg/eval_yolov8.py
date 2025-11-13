import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms

from dataset import isolated_dataset_loader

val_transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


ds = isolated_dataset_loader(batch_size=32, transforms=val_transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_wrapper = YOLO("yolov8n-cls.pt")
model = yolo_wrapper.model

yolo_wrapper = YOLO("yolov8n-cls.pt")
model = yolo_wrapper.model

for name, module in model.named_modules():
    if name == "model.9.linear":
        in_features = module.in_features
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        layer_name = name.rsplit(".", 1)[-1] if "." in name else name

        if parent_name:
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model

        new_layer = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 1)).to(
            device
        )

        setattr(parent, layer_name, new_layer)
        break

model = model.to(device)

model.load_state_dict(
    torch.load("models/yolov8_piano_key/model.pth", map_location=device)
)

model.eval()


def evaluate(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)[0]
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


accuracy = evaluate(ds)
print(accuracy)
