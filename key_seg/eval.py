import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the same model architecture
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # binary classifier

# Load weights
model.load_state_dict(torch.load("models/resnet_piano_key.pth"))
model = model.to(device)
model.eval()  # set to evaluation mode

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_path = "infer.png"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # add batch dimension
with torch.no_grad():
    output = model(image)  # raw logits
    prob = torch.sigmoid(output)  # probability 0-1
    pred = (prob > 0.5).int()  # binary prediction

print(f"Probability pressed: {prob.item():.3f}")
print(f"Predicted label: {pred.item()}")
