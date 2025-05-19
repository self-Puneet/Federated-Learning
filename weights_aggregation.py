import numpy as np
import os
from scipy.spatial.distance import cdist

def load_layer_weights(model_dirs, layer_key):
    """
    Searches for a .npy file that contains the layer_key as substring.
    Returns list of loaded arrays or None if not found.
    """
    weights = []
    for model_dir in model_dirs:
        found = False
        for fname in os.listdir(model_dir):
            if layer_key in fname and fname.endswith(".npy"):
                full_path = os.path.join(model_dir, fname)
                weights.append(np.load(full_path))
                found = True
                break
        if not found:
            print(f"⚠️ '{layer_key}' not found in: {model_dir}")
            weights.append(None)
    return weights

def match_and_average_neurons(weight_matrices, similarity='euclidean', threshold=None):
    """
    Aggregates neuron matrices layer-wise using similarity matching.
    Returns a [num_global_neurons, dim] matrix or None if nothing was matched.
    """
    all_neurons = []
    for model_id, mat in enumerate(weight_matrices):
        if mat is None:
            continue
        for i in range(mat.shape[0]):
            all_neurons.append((model_id, i, mat[i]))

    used = set()
    aggregated = []

    while True:
        unused = [(i, nid, vec) for (i, nid, vec) in all_neurons if (i, nid) not in used]
        if len(unused) < 2:
            for (_, _, vec) in unused:
                aggregated.append(vec)
            break

        best_pair = None
        best_score = float('inf')

        for i in range(len(unused)):
            for j in range(i + 1, len(unused)):
                m1, id1, v1 = unused[i]
                m2, id2, v2 = unused[j]
                if m1 == m2:
                    continue
                if v1.shape != v2.shape:
                    continue
                score = np.linalg.norm(v1 - v2) if similarity == 'euclidean' else 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if score < best_score:
                    best_score = score
                    best_pair = ((m1, id1, v1), (m2, id2, v2))

        if best_pair is None:
            print("⚠️ No more compatible neurons to match.")
            break

        (m1, id1, v1), (m2, id2, v2) = best_pair
        used.add((m1, id1))
        used.add((m2, id2))
        aggregated.append((v1 + v2) / 2)

    if len(aggregated) == 0:
        print("❌ No neurons were aggregated. Check input weight dimensions.")
        return None

    return np.vstack(aggregated)

def aggregate_layers(model_dirs, layer_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for layer in layer_names:
        print(f"\n🔄 Aggregating layer: {layer}")
        weights = load_layer_weights(model_dirs, layer)
        valid_weights = [w for w in weights if w is not None]
        if len(valid_weights) < 2:
            print("❌ Not enough models with valid weights. Skipping.")
            continue
        aggregated = match_and_average_neurons(valid_weights)
        if aggregated is None:
            print(f"⚠️ Aggregation failed for layer: {layer}")
            continue
        out_path = os.path.join(output_dir, f"{layer}.npy")
        np.save(out_path, aggregated)
        print(f"✅ Saved aggregated layer to: {out_path}")

if __name__ == "__main__":
    model_dirs = [
        r"image_model_weights\epoch_4",
        r"lstm_model_weights_fixed\epoch_10"
    ]

    output_dir = r"aggregated_weights"

    layer_names = [
        "projector_weight", "projector_bias",
        "classifier_weight", "classifier_bias"
    ]

    aggregate_layers(model_dirs, layer_names, output_dir)