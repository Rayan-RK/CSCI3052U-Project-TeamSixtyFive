
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# Load predictions from the Random Forest baseline
data = pd.read_csv(OUTPUTS / "rgb_predictions.csv")

classes = sorted(
    set(data["true_class"]) | set(data["predicted_class"])
)

for partition, filename in [
    ("development_validation", "confusion_matrix_dev.png"),
    ("ood_test", "confusion_matrix_ood.png"),
]:
    subset = data[data["partition"] == partition]

    actual = subset["true_class"]
    predicted = subset["predicted_class"]

    # Create confusion matrix
    cm = confusion_matrix(
        actual,
        predicted,
        labels=classes
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    ax.set_title(f"Confusion Matrix - {partition}")
    plt.tight_layout()

    # Save graph
    plt.savefig(OUTPUTS / filename, dpi=200)
    plt.close()

    # Identify incorrect predictions
    mistakes = subset[
        subset["true_class"] != subset["predicted_class"]
    ]

    top_errors = (
        mistakes
        .groupby(["true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\nResults for {partition}")
    print(f"Total samples: {len(subset)}")
    print(f"Incorrect predictions: {len(mistakes)}")

    print("\nTop 5 classification mistakes:")
    print(top_errors.head(5).to_string(index=False))

    top_errors.to_csv(
        OUTPUTS / f"errors_{partition}.csv",
        index=False
    )

print("\nError analysis complete.")
