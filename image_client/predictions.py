import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import pandas as pd

# --- Config ---
TEST_ROOT = r"data\image_database"  # Root folder with subfolders of classes
MODEL_PATH = r"global_model.pth"
OUTPUT_CSV = r"image_client\prediction.csv"

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Feature extractor: ResNet18 (pretrained, up to penultimate layer) ---
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
resnet.eval()

# --- Load your global model ---
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = torch.nn.Linear(512, 128)
        self.classifier = torch.nn.Linear(128, 3)

    def forward(self, x):
        z = self.projector(x)
        return self.classifier(z)

model = GlobalModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# --- Class mapping ---
class_names = ["d_Unhealthy", "e_Very_Unhealthy", "f_Severe"]
class_to_index = {name: i for i, name in enumerate(class_names)}

# --- Predict function ---
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = resnet(x).view(1, -1)
        logits = model(feat)
        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).squeeze()[pred_idx].item()
        return class_names[pred_idx], confidence

# --- Walk all test images ---
results = []

for class_folder in class_names:
    class_path = os.path.join(TEST_ROOT, class_folder)
    if not os.path.isdir(class_path):
        print(f"⚠️ Folder not found: {class_path}")
        continue

    for fname in os.listdir(class_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        full_path = os.path.join(class_path, fname)
        pred_label, conf = predict_image(full_path)
        results.append({
            'file': fname,
            'true_label': class_folder,
            'predicted_label': pred_label,
            'confidence': round(conf, 4)
        })
        print(f"{fname}: ✅ Pred = {pred_label} ({conf:.2f}) | True = {class_folder}")

# --- Save results ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ All predictions saved to: {OUTPUT_CSV}")
