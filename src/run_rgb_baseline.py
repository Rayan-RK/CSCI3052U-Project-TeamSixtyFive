from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
IMAGE_SIZE = (16, 16)
PCA_COMPONENTS = 100
N_TREES = 200


def main():
    project_root = Path(__file__).resolve().parents[1]

    manifest_path = project_root / "outputs" / "geographic_manifest_rebuilt.csv"
    rgb_root = project_root / "data" / "EuroSAT_RGB" / "EuroSAT_RGB"
    output_path = project_root / "outputs" / "rgb_baseline_results.csv"

    manifest = pd.read_csv(manifest_path)

    data = manifest[
        manifest["candidate_partition"].isin(["development", "ood_test"])
    ].copy()

    image_paths = {
        path.stem: path
        for class_folder in rgb_root.iterdir()
        if class_folder.is_dir()
        for path in class_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    data = data[data["sample_id"].isin(image_paths)].copy()

    if data.empty:
        raise ValueError("No matching images found. Check the RGB folder path.")


    features = []
    for sample_id in data["sample_id"]:
        with Image.open(image_paths[sample_id]) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            features.append(np.asarray(image, dtype=np.float32).flatten() / 255.0)

    X = np.asarray(features, dtype=np.float32)
    y = data["class"].to_numpy()
    partitions = data["candidate_partition"].to_numpy()

    dev_mask = partitions == "development"
    ood_mask = partitions == "ood_test"

    X_dev, y_dev = X[dev_mask], y[dev_mask]
    X_ood, y_ood = X[ood_mask], y[ood_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_dev,
    )

    model = Pipeline([
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)),
        ("random_forest", RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print(f"Development training images: {len(y_train)}")
    print(f"Development validation images: {len(y_val)}")
    print(f"OOD test images: {len(y_ood)}")
    print("Training PCA + Random Forest...")

    model.fit(X_train, y_train)


    results = []
    class_results = []

    for name, X_eval, y_eval in [
        ("development_validation", X_val, y_val),
        ("ood_test", X_ood, y_ood),
    ]:
        predictions = model.predict(X_eval)

        results.append({
            "model": "PCA + Random Forest (RGB)",
            "partition": name,
            "n_samples": len(y_eval),
            "accuracy": accuracy_score(y_eval, predictions),
            "macro_f1": f1_score(
                y_eval, predictions, average="macro", zero_division=0
            ),
        })

        report = classification_report(
            y_eval,
            predictions,
            labels=sorted(np.unique(y)),
            output_dict=True,
            zero_division=0,
        )

        for class_name in sorted(np.unique(y)):
            metrics = report[class_name]
            class_results.append({
                "model": "PCA + Random Forest (RGB)",
                "partition": name,
                "class": class_name,
                "support": metrics["support"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    class_output_path = (
        project_root / "outputs" / "rgb_per_class_results.csv"
    )
    class_results_df = pd.DataFrame(class_results)
    class_results_df.to_csv(class_output_path, index=False)

    print("\nOverall results:")
    print(results_df.to_string(index=False))

    print("\nPer-class results:")
    print(class_results_df.round(3).to_string(index=False))

    print(f"\nSaved overall results to: {output_path}")
    print(f"Saved per-class results to: {class_output_path}")



if __name__ == "__main__":
    main()
