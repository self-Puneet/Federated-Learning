import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

def evaluate_model(csv_path):
    # Load prediction CSV
    df = pd.read_csv(csv_path)

    # Optional: Clean column names
    df.columns = df.columns.str.strip()

    # Extract labels
    true = df['true_label'].astype(str)
    pred = df['predicted_label'].astype(str)

    # Print basic metrics
    print("📊 Classification Report:\n")
    print(classification_report(true, pred, digits=4))

    acc = accuracy_score(true, pred)
    print(f"✅ Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true, pred, labels=sorted(true.unique()))
    labels = sorted(true.unique())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("🧾 Confusion Matrix - Image Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# Replace with your actual CSV path
image_csv_path = "image_client\prediction.csv"
evaluate_model(image_csv_path)
