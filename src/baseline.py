
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def main():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "outputs" / "geographic_manifest_rebuilt.csv"
    output_path = project_root / "outputs" / "baseline_results.csv"

    manifest = pd.read_csv(manifest_path)

    development = manifest[
        manifest["candidate_partition"] == "development"
    ].copy()

    ood_test = manifest[
        manifest["candidate_partition"] == "ood_test"
    ].copy()

    train, validation = train_test_split(
        development,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=development["class"],
    )

    X_train = np.zeros((len(train), 1))
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, train["class"])

    results = []

    for name, partition in [
        ("development_validation", validation),
        ("ood_test", ood_test),
    ]:
        X_eval = np.zeros((len(partition), 1))
        predictions = model.predict(X_eval)
        actual = partition["class"]

        results.append({
            "model": "Majority-class baseline",
            "partition": name,
            "n_samples": len(partition),
            "accuracy": accuracy_score(actual, predictions),
            "macro_f1": f1_score(
                actual, predictions, average="macro", zero_division=0
            ),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print("Predicted class:", model.classes_[model.class_prior_.argmax()])
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
