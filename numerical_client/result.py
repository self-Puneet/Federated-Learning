import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
# final_model.pth -> image classify -> 30% (multiple classes prediction), final_model.pth -> numerical classify -> 5%
def evaluate_model(csv_path):
    # Load prediction CSV
    df = pd.read_csv(csv_path)

    # Optional: Clean column names
    df.columns = df.columns.str.strip()

    # Extract labels
    true = df['True_Label'].astype(str)
    pred = df['Predicted_Label'].astype(str)

    # Print classification metrics
    print("📊 Classification Report:\n")
    print(classification_report(true, pred, digits=4))

    acc = accuracy_score(true, pred)
    print(f"✅ Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true, pred, labels=sorted(true.unique()))
    labels = sorted(true.unique())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title("🧾 Confusion Matrix - Numerical Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# Replace with your actual CSV path
numerical_csv_path = "numerical_client\prediction.csv"
evaluate_model(numerical_csv_path)
