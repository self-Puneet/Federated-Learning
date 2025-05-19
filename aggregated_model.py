import torch
import torch.nn as nn
import numpy as np

AGG_DIR = r"aggregated_weights"
EMBED_DIM = 128
NUM_CLASSES = 3  # d_Unhealthy, e_Very_Unhealthy, f_Severe

class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(512, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        z = self.projector(x)
        return self.classifier(z)

# Load model
model = GlobalModel()

# Load and reshape .npy weights if needed
def load_and_fix_bias(path):
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(1)  # Convert from [N, 1] to [N]
    return arr

with torch.no_grad():
    model.projector.weight.copy_(torch.tensor(np.load(f"{AGG_DIR}/projector_weight.npy")))
    model.projector.bias.copy_(torch.tensor(load_and_fix_bias(f"{AGG_DIR}/projector_bias.npy")))
    model.classifier.weight.copy_(torch.tensor(np.load(f"{AGG_DIR}/classifier_weight.npy")))
    model.classifier.bias.copy_(torch.tensor(load_and_fix_bias(f"{AGG_DIR}/classifier_bias.npy")))

# Save
torch.save(model.state_dict(), "global_model.pth")
print("✅ Global model saved as global_model.pth")
