import numpy as np
import os

AGG_DIR = r"aggregated_weights"

for fname in os.listdir(AGG_DIR):
    if fname.endswith(".npy"):
        arr = np.load(os.path.join(AGG_DIR, fname))
        print(f"{fname}: shape = {arr.shape}")
