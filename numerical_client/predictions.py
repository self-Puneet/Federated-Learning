import pandas as pd
import torch
import torch.nn as nn
import os

# --- Paths ---
CSV_PATH = r"data\numerical_database.csv"
MODEL_PATH = r"global_model.pth"
OUTPUT_CSV = r"numerical_client\prediction.csv"

# --- Define the same LSTMEncoder used during training ---
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden]
        final_out = out[:, -1, :]  # Use last time step
        return self.fc(final_out)

# --- Global model (projector + classifier) ---
class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, 3)

    def forward(self, x):
        z = self.projector(x)
        return self.classifier(z)

# --- Load models ---
encoder = LSTMEncoder()
global_model = GlobalModel()
global_model.load_state_dict(torch.load(MODEL_PATH))
encoder.eval()
global_model.eval()

# --- Class labels ---
label_map = ['d_Unhealthy', 'e_Very_Unhealthy', 'f_Severe']
label_to_index = {label: i for i, label in enumerate(label_map)}

# --- Load and preprocess data ---
df = pd.read_csv(CSV_PATH)
numerical_features = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

results = []

for _, row in df.iterrows():
    # data_seq = torch.tensor(row[numerical_features].values, dtype=torch.float32).view(1, 1, -1)  # [batch, seq_len, features]
    values = [float(row[f]) for f in numerical_features]
    data_seq = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
        
    with torch.no_grad():
        encoded_feat = encoder(data_seq)  # [1, 512]
        logits = global_model(encoded_feat)
        pred_idx = torch.argmax(logits, dim=1).item()
        conf = torch.softmax(logits, dim=1).squeeze()[pred_idx].item()

    results.append({
        'Filename': row['Filename'],
        'True_Label': row['AQI_Class'],
        'Predicted_Label': label_map[pred_idx],
        'Confidence': round(conf, 4)
    })

# --- Save results ---
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Numerical classification results saved to: {OUTPUT_CSV}")
