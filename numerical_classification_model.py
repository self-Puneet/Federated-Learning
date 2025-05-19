# fixed_lstm_model.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Config ---
CSV_PATH = "/content/drive/MyDrive/numerical_database.csv"
SAVE_DIR = "/content/drive/MyDrive/lstm_model_weights_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
HIDDEN_DIM = 512
EMBED_DIM = 128
NUM_FEATURES = 7

class AQIDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        print("Original rows:", len(df))

        df = df.dropna()
        print("After dropna:", len(df))

        features = ['AQI', 'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
        
        # Convert only feature columns
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=features)
        print("After numeric filtering:", len(df))

        X = StandardScaler().fit_transform(df[features].values)
        le = LabelEncoder()
        y = le.fit_transform(df['AQI_Class'])

        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        print(f"✅ Final dataset: {len(self.X)} samples, Classes: {le.classes_}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model ---
class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(NUM_FEATURES, HIDDEN_DIM, batch_first=True, dropout=0.3)
        self.projector = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.projector(h_n[-1])
        out = self.classifier(z)
        return z, out

# --- Save weights ---
def save_layer_weights(model, save_dir, epoch):
    layer_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
    os.makedirs(layer_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name.replace('.', '_')
            np.save(os.path.join(layer_dir, f"{clean_name}.npy"), param.detach().cpu().numpy())
    print(f"✅ Saved weights to {layer_dir}")

# --- Train ---
def train(model, loader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss):
                print("❌ NaN loss detected. Skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"lstm_epoch_{epoch+1}.pth"))
        save_layer_weights(model, SAVE_DIR, epoch)

# --- Run ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = AQIDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMModel(num_classes=len(set(dataset.y.tolist()))).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train(model, loader, optimizer, criterion)
