# image_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

# --- Config ---
DATA_DIR = "/content/drive/MyDrive/image_database"
SAVE_DIR = "/content/drive/MyDrive/image_model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 4
LR = 1e-4
EMBED_DIM = 128
ENCODER_DIM = 512  # Match with LSTM

# --- Model ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Remove FC
        self.projector = nn.Linear(ENCODER_DIM, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        z = self.projector(x)
        out = self.classifier(z)
        return z, out

def save_layer_weights(model, save_dir, epoch):
    layer_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
    os.makedirs(layer_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name.replace('.', '_')
            npy_path = os.path.join(layer_dir, f"{clean_name}.npy")
            torch_np = param.detach().cpu().numpy()
            np.save(npy_path, torch_np)
    print(f"✅ Saved weights to {layer_dir}")

def train(model, loader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"{SAVE_DIR}/cnn_epoch_{epoch+1}.pth")

        # ✅ Also save individual layer weights
        save_layer_weights(model, SAVE_DIR, epoch)

        
# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNModel(num_classes=len(dataset.classes)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
train(model, loader, optimizer, criterion)
