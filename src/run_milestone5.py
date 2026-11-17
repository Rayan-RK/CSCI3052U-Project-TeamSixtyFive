# Milestone 5 evaluation script for CSCI 3052U Project (Team Sixty-Five)
# Runs hyperparameter tuning, 5-seed evaluation, error analysis, sensitivity/robustness checks, and ablations

import copy
import json
import os
import platform
import random
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend so plots save cleanly without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# Set up project paths and random seeds
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "outputs" / "geographic_manifest_rebuilt.csv"
RGB_ROOT = ROOT / "data" / "EuroSAT_RGB" / "EuroSAT_RGB"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 100, 2026, 777, 999]
SPLIT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Grab the current git commit hash so we can log which code version produced the results
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# Lock all random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Download and unzip EuroSAT_RGB automatically if it isn't in the data folder yet
def ensure_eurosat_rgb():
    if RGB_ROOT.exists() and any(RGB_ROOT.iterdir()):
        return
    print("EuroSAT_RGB not found locally. Downloading (~90 MB)...")
    zip_path = ROOT / "data" / "EuroSAT.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting EuroSAT.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ROOT / "data" / "EuroSAT_RGB")
    nested = ROOT / "data" / "EuroSAT_RGB" / "2750"
    if nested.exists() and not RGB_ROOT.exists():
        nested.rename(RGB_ROOT)
    zip_path.unlink(missing_ok=True)
    print(f"Dataset ready at {RGB_ROOT}")


# Custom PyTorch Dataset to load satellite JPGs from disk on the fly
class EuroSATDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[idx]


# Load ResNet-18 and replace the final 1000-class layer with our 10 land-cover classes
def build_resnet18(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# Set up image resizing, data augmentation (flips + rotations), and ImageNet normalization
def get_transforms(img_size: int = 64, augment: bool = True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_ops = [transforms.Resize((img_size, img_size))]
    if augment:
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    eval_ops = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


# Train ResNet-18 on the dev train split, save the best checkpoint based on val Macro-F1, and evaluate
def train_and_eval_resnet(
    train_df,
    val_df,
    ood_df,
    class_to_idx,
    seed: int,
    lr: float = 3e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    epochs: int = 12,
    pretrained: bool = True,
    augment: bool = True,
    class_weighted: bool = True,
    img_size: int = 64,
    eval_ood: bool = False,
):
    set_seed(seed)
    t0 = time.time()
    train_tf, eval_tf = get_transforms(img_size=img_size, augment=augment)

    train_ds = EuroSATDataset(train_df["path"], [class_to_idx[c] for c in train_df["class"]], train_tf)
    val_ds = EuroSATDataset(val_df["path"], [class_to_idx[c] for c in val_df["class"]], eval_tf)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin)

    model = build_resnet18(len(class_to_idx), pretrained=pretrained)

    # Use inverse class frequencies as loss weights so smaller classes aren't ignored
    if class_weighted:
        counts = np.bincount([class_to_idx[c] for c in train_df["class"]], minlength=len(class_to_idx))
        weights = torch.tensor(len(train_df) / (len(class_to_idx) * counts), dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Check validation Macro-F1 after each epoch and keep the best weights
        val_preds, val_true, _ = predict_resnet(model, val_loader)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    # Reload the best validation model before computing final metrics
    model.load_state_dict(best_weights)
    val_preds, val_true, _ = predict_resnet(model, val_loader)
    val_acc = accuracy_score(val_true, val_preds)
    val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

    # Only run on OOD test when explicitly requested (never during hyperparameter tuning!)
    ood_acc, ood_f1, ood_preds, ood_true, ood_probs = None, None, None, None, None
    if eval_ood:
        ood_ds = EuroSATDataset(ood_df["path"], [class_to_idx[c] for c in ood_df["class"]], eval_tf)
        ood_loader = DataLoader(ood_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin)
        ood_preds, ood_true, ood_probs = predict_resnet(model, ood_loader)
        ood_acc = accuracy_score(ood_true, ood_preds)
        ood_f1 = f1_score(ood_true, ood_preds, average="macro", zero_division=0)

    runtime_min = (time.time() - t0) / 60.0
    return {
        "model": model,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "ood_acc": ood_acc,
        "ood_f1": ood_f1,
        "val_preds": val_preds,
        "val_true": val_true,
        "ood_preds": ood_preds,
        "ood_true": ood_true,
        "ood_probs": ood_probs,
        "runtime_min": runtime_min,
    }


# Helper function to get predicted classes and softmax probabilities without tracking gradients
@torch.no_grad()
def predict_resnet(model, loader):
    model.eval()
    preds, trues, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(probs)


# Resize images to 16x16 and flatten RGB channels into 768-element vectors for Random Forest
# Also supports adding noise or brightness/contrast shifts for our robustness tests
def extract_flat_rgb(df, size=(16, 16), corrupt_mode=None, seed=42):
    rng = np.random.default_rng(seed)
    feats = []
    for p in df["path"]:
        with Image.open(p) as img:
            img = img.convert("RGB")
            if corrupt_mode == "brightness":
                img = ImageEnhance.Brightness(img).enhance(1.35)
                img = ImageEnhance.Contrast(img).enhance(1.25)
            img = img.resize(size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if corrupt_mode == "noise":
                arr = np.clip(arr + rng.normal(0, 0.08, arr.shape).astype(np.float32), 0.0, 1.0)
            feats.append(arr.flatten())
    return np.asarray(feats, dtype=np.float32)


# Calculate mean, sample SD, and 95% Student-t confidence interval across the 5 seeds
def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    mean = arr.mean()
    if len(arr) < 2 or np.std(arr, ddof=1) == 0:
        return mean, 0.0, mean, mean
    sd = np.std(arr, ddof=1)
    h = stats.t.ppf(0.975, df=len(arr) - 1) * (sd / np.sqrt(len(arr)))
    return mean, sd, mean - h, mean + h


# Resample test predictions 1000 times with replacement to get a 95% bootstrap CI
def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    ensure_eurosat_rgb()

    # Load our geographic manifest and filter out unlocalized 'NoData' images
    manifest = pd.read_csv(MANIFEST_PATH)
    data = manifest[manifest["candidate_partition"].isin(["development", "ood_test"])].copy()

    # Map each sample_id to its actual image path on disk
    image_paths = {
        path.stem: path
        for folder in RGB_ROOT.iterdir()
        if folder.is_dir()
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    data["path"] = data["sample_id"].map(image_paths)
    data = data.dropna(subset=["path"]).reset_index(drop=True)

    classes = sorted(data["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Keep Development (North/West/Central Europe) completely separate from OOD Test (South/East Europe)
    dev_df = data[data["candidate_partition"] == "development"].reset_index(drop=True)
    ood_df = data[data["candidate_partition"] == "ood_test"].reset_index(drop=True)

    # Split Development 80/20 into train and validation using a fixed seed
    train_df, val_df = train_test_split(
        dev_df, test_size=0.2, random_state=SPLIT_SEED, stratify=dev_df["class"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Device: {DEVICE} | Train: {len(train_df)} | Val: {len(val_df)} | OOD Test: {len(ood_df)}")

    # Helper to record every experiment run with a unique Run ID (E001, E002, ...)
    exp_log = []
    run_counter = 1

    def log_run(model_name, hyperparams, seed, val_f1, ood_f1, runtime_min, obs):
        nonlocal run_counter
        rid = f"E{run_counter:03d}"
        run_counter += 1
        exp_log.append({
            "run_id": rid,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": model_name,
            "hyperparameters": hyperparams,
            "seed": seed,
            "split": "Dev 80/20 (seed 42) vs Geographic OOD",
            "val_macro_f1": round(val_f1, 4) if val_f1 is not None else "",
            "ood_test_macro_f1": round(ood_f1, 4) if ood_f1 is not None else "",
            "runtime_min": round(runtime_min, 2),
            "observations": obs,
        })
        return rid

    # Pre-extract 16x16 flattened features once so Random Forest runs fast
    X_train_rf = extract_flat_rgb(train_df)
    X_val_rf = extract_flat_rgb(val_df)
    X_ood_rf = extract_flat_rgb(ood_df)
    y_train = train_df["class"].to_numpy()
    y_val = val_df["class"].to_numpy()
    y_ood = ood_df["class"].to_numpy()

  
    # 1. HYPERPARAMETER TUNING (12 trials RF, 12 trials ResNet-18)
    print("\n--- Step 1: Hyperparameter Tuning on Dev Validation ---")
    tuning_rows = []

    # Sample 12 random combinations for PCA + Random Forest
    rf_grid = [
        {"n_estimators": n, "max_depth": d, "max_features": mf}
        for n in [100, 200, 400]
        for d in [15, 25, None]
        for mf in ["sqrt", "log2"]
    ]
    rng = random.Random(SPLIT_SEED)
    rf_trials = rng.sample(rf_grid, 12)

    best_rf_cfg = None
    best_rf_val = -1.0
    for cfg in rf_trials:
        t0 = time.time()
        pipe = Pipeline([
            ("pca", PCA(n_components=100, random_state=SPLIT_SEED)),
            ("rf", RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                max_features=cfg["max_features"],
                class_weight="balanced",
                random_state=SPLIT_SEED,
                n_jobs=-1,
            )),
        ])
        pipe.fit(X_train_rf, y_train)
        preds = pipe.predict(X_val_rf)
        vf1 = f1_score(y_val, preds, average="macro", zero_division=0)
        rt = (time.time() - t0) / 60.0
        hp_str = f"pca=100, n_trees={cfg['n_estimators']}, max_depth={cfg['max_depth']}, max_feat={cfg['max_features']}"
        rid = log_run("PCA + Random Forest", hp_str, SPLIT_SEED, vf1, None, rt, "RF tuning trial")
        tuning_rows.append({"run_id": rid, "model": "PCA + Random Forest", **cfg, "val_macro_f1": vf1})
        if vf1 > best_rf_val:
            best_rf_val = vf1
            best_rf_cfg = cfg

    # Sample 12 random combinations for ResNet-18 (same budget as RF)
    resnet_grid = [
        {"lr": lr, "batch_size": bs, "weight_decay": wd, "epochs": 10}
        for lr in [1e-4, 3e-4, 1e-3, 3e-3]
        for bs in [32, 64]
        for wd in [1e-5, 1e-4]
    ]
    resnet_trials = rng.sample(resnet_grid, 12)

    best_rn_cfg = None
    best_rn_val = -1.0
    for cfg in resnet_trials:
        res = train_and_eval_resnet(
            train_df, val_df, ood_df, class_to_idx, seed=SPLIT_SEED,
            lr=cfg["lr"], batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
            epochs=cfg["epochs"], eval_ood=False
        )
        hp_str = f"lr={cfg['lr']}, bs={cfg['batch_size']}, wd={cfg['weight_decay']}, epochs={cfg['epochs']}"
        rid = log_run("ResNet-18 (Proposed)", hp_str, SPLIT_SEED, res["val_f1"], None, res["runtime_min"], "ResNet-18 tuning trial")
        tuning_rows.append({"run_id": rid, "model": "ResNet-18 (Proposed)", **cfg, "val_macro_f1": res["val_f1"]})
        if res["val_f1"] > best_rn_val:
            best_rn_val = res["val_f1"]
            best_rn_cfg = cfg

    pd.DataFrame(tuning_rows).to_csv(OUT_DIR / "m5_tuning_records.csv", index=False)
    print(f"Best RF Config: {best_rf_cfg} (Val Macro-F1={best_rf_val:.4f})")
    print(f"Best ResNet-18 Config: {best_rn_cfg} (Val Macro-F1={best_rn_val:.4f})")


    # 2. 5-SEED FINAL EVALUATION & UNCERTAINTY
    print("\n--- Step 2: 5-Seed Evaluation on Val & OOD Test ---")
    per_seed_rows = []
    seed42_artifacts = {}

    for s in SEEDS:
        # Baseline 1: Majority-class dummy classifier
        t0 = time.time()
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_df), 1)), y_train)
        vp = dummy.predict(np.zeros((len(val_df), 1)))
        op = dummy.predict(np.zeros((len(ood_df), 1)))
        rt = (time.time() - t0) / 60.0
        va, vf = accuracy_score(y_val, vp), f1_score(y_val, vp, average="macro", zero_division=0)
        oa, of = accuracy_score(y_ood, op), f1_score(y_ood, op, average="macro", zero_division=0)
        rid = log_run("Majority-Class Baseline", "strategy=most_frequent", s, vf, of, rt, f"Seed {s} final eval")
        per_seed_rows.append({"run_id": rid, "model": "Majority-Class Baseline", "seed": s, "val_acc": va, "val_f1": vf, "ood_acc": oa, "ood_f1": of, "runtime_min": rt})

        # Baseline 2: Tuned PCA + Random Forest
        t0 = time.time()
        rf_pipe = Pipeline([
            ("pca", PCA(n_components=100, random_state=s)),
            ("rf", RandomForestClassifier(
                n_estimators=best_rf_cfg["n_estimators"],
                max_depth=best_rf_cfg["max_depth"],
                max_features=best_rf_cfg["max_features"],
                class_weight="balanced",
                random_state=s,
                n_jobs=-1,
            )),
        ])
        rf_pipe.fit(X_train_rf, y_train)
        vp_rf = rf_pipe.predict(X_val_rf)
        op_rf = rf_pipe.predict(X_ood_rf)
        rt = (time.time() - t0) / 60.0
        va, vf = accuracy_score(y_val, vp_rf), f1_score(y_val, vp_rf, average="macro", zero_division=0)
        oa, of = accuracy_score(y_ood, op_rf), f1_score(y_ood, op_rf, average="macro", zero_division=0)
        rid = log_run("PCA + Random Forest", str(best_rf_cfg), s, vf, of, rt, f"Seed {s} final eval")
        per_seed_rows.append({"run_id": rid, "model": "PCA + Random Forest", "seed": s, "val_acc": va, "val_f1": vf, "ood_acc": oa, "ood_f1": of, "runtime_min": rt})
        if s == 42:
            seed42_artifacts["rf_pipe"] = rf_pipe
            seed42_artifacts["rf_ood_preds"] = op_rf

        # Proposed Model: Tuned ResNet-18 trained for 15 epochs
        res = train_and_eval_resnet(
            train_df, val_df, ood_df, class_to_idx, seed=s,
            lr=best_rn_cfg["lr"], batch_size=best_rn_cfg["batch_size"],
            weight_decay=best_rn_cfg["weight_decay"], epochs=15, eval_ood=True
        )
        rid = log_run("ResNet-18 (Proposed)", f"{best_rn_cfg}, epochs=15", s, res["val_f1"], res["ood_f1"], res["runtime_min"], f"Seed {s} final eval")
        per_seed_rows.append({
            "run_id": rid, "model": "ResNet-18 (Proposed)", "seed": s,
            "val_acc": res["val_acc"], "val_f1": res["val_f1"],
            "ood_acc": res["ood_acc"], "ood_f1": res["ood_f1"],
            "runtime_min": res["runtime_min"]
        })
        # Save seed 42 models and predictions for confusion matrices, error analysis, and ablations
        if s == 42:
            seed42_artifacts["rn_model"] = res["model"]
            seed42_artifacts["rn_val_preds"] = res["val_preds"]
            seed42_artifacts["rn_val_true"] = res["val_true"]
            seed42_artifacts["rn_ood_preds"] = res["ood_preds"]
            seed42_artifacts["rn_ood_true"] = res["ood_true"]
            seed42_artifacts["rn_ood_probs"] = res["ood_probs"]

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT_DIR / "m5_raw_per_seed.csv", index=False)

    # Aggregate 5-seed means, SDs, and 95% confidence intervals into our main summary table
    summary_rows = []
    for m_name in ["Majority-Class Baseline", "PCA + Random Forest", "ResNet-18 (Proposed)"]:
        sub = per_seed_df[per_seed_df["model"] == m_name]
        va_m, va_sd, va_lo, va_hi = ci95(sub["val_acc"])
        vf_m, vf_sd, vf_lo, vf_hi = ci95(sub["val_f1"])
        oa_m, oa_sd, oa_lo, oa_hi = ci95(sub["ood_acc"])
        of_m, of_sd, of_lo, of_hi = ci95(sub["ood_f1"])
        summary_rows.append({
            "model": m_name,
            "val_acc": f"{va_m:.4f} ± {va_sd:.4f} [{va_lo:.4f}, {va_hi:.4f}]",
            "val_macro_f1": f"{vf_m:.4f} ± {vf_sd:.4f} [{vf_lo:.4f}, {vf_hi:.4f}]",
            "ood_acc": f"{oa_m:.4f} ± {oa_sd:.4f} [{oa_lo:.4f}, {oa_hi:.4f}]",
            "ood_macro_f1": f"{of_m:.4f} ± {of_sd:.4f} [{of_lo:.4f}, {of_hi:.4f}]",
            "ood_f1_mean": of_m,
            "ood_f1_ci_half": (of_hi - of_lo) / 2.0,
            "val_f1_mean": vf_m,
            "val_f1_ci_half": (vf_hi - vf_lo) / 2.0,
            "avg_train_min": round(sub["runtime_min"].mean(), 2),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "m5_main_results.csv", index=False)
    print("\nMain Table (5 Seeds):")
    print(summary_df[["model", "val_macro_f1", "ood_macro_f1", "avg_train_min"]].to_string(index=False))

    # Run Wilcoxon signed-rank tests across seeds and bootstrap CI on seed 42 OOD predictions
    rn_ood_scores = per_seed_df[per_seed_df["model"] == "ResNet-18 (Proposed)"]["ood_f1"].to_numpy()
    rf_ood_scores = per_seed_df[per_seed_df["model"] == "PCA + Random Forest"]["ood_f1"].to_numpy()
    maj_ood_scores = per_seed_df[per_seed_df["model"] == "Majority-Class Baseline"]["ood_f1"].to_numpy()

    _, p_rn_rf = stats.wilcoxon(rn_ood_scores, rf_ood_scores)
    _, p_rn_maj = stats.wilcoxon(rn_ood_scores, maj_ood_scores)
    boot_lo, boot_hi = bootstrap_ci(seed42_artifacts["rn_ood_true"], seed42_artifacts["rn_ood_preds"])

    unc_stats = {
        "wilcoxon_resnet_vs_rf_p": float(p_rn_rf),
        "wilcoxon_resnet_vs_majority_p": float(p_rn_maj),
        "resnet_seed42_ood_bootstrap_95ci": [boot_lo, boot_hi],
    }
    with open(OUT_DIR / "m5_uncertainty_stats.json", "w") as f:
        json.dump(unc_stats, f, indent=2)

    # Plot Figure 1: Bar chart comparing Dev Validation vs Geographic OOD Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    w = 0.35
    ax.bar(x - w/2, summary_df["val_f1_mean"], w, yerr=summary_df["val_f1_ci_half"], capsize=5, label="Dev Validation", color="#4C72B0")
    ax.bar(x + w/2, summary_df["ood_f1_mean"], w, yerr=summary_df["ood_f1_ci_half"], capsize=5, label="Geographic OOD Test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
    ax.set_ylabel("Macro-F1 (Mean ± 95% CI)")
    ax.set_title("Development Validation vs. Geographic OOD Macro-F1 Across Models")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "m5_fig1_model_comparison_ci.png", dpi=300)
    plt.close()

    # 3. ERROR ANALYSIS & CONFUSION MATRICES

    print("\n--- Step 3: Error Analysis & Confusion Matrices ---")
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ood_true_names = [idx_to_class[i] for i in seed42_artifacts["rn_ood_true"]]
    ood_pred_names = [idx_to_class[i] for i in seed42_artifacts["rn_ood_preds"]]
    val_true_names = [idx_to_class[i] for i in seed42_artifacts["rn_val_true"]]
    val_pred_names = [idx_to_class[i] for i in seed42_artifacts["rn_val_preds"]]

    # Compute per-class F1 scores and rank classes by how much they drop on OOD test
    rep_val = classification_report(val_true_names, val_pred_names, labels=classes, output_dict=True, zero_division=0)
    rep_ood = classification_report(ood_true_names, ood_pred_names, labels=classes, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame([
        {
            "class": c,
            "dev_val_f1": round(rep_val[c]["f1-score"], 4),
            "ood_test_f1": round(rep_ood[c]["f1-score"], 4),
            "f1_drop": round(rep_val[c]["f1-score"] - rep_ood[c]["f1-score"], 4),
            "ood_support": int(rep_ood[c]["support"]),
        }
        for c in classes
    ]).sort_values("f1_drop", ascending=False)
    per_class_df.to_csv(OUT_DIR / "m5_per_class_f1.csv", index=False)

    # Save the 50 highest-confidence mistakes on OOD test so we can inspect failure cases
    probs = seed42_artifacts["rn_ood_probs"]
    err_df = ood_df.copy()
    err_df["true_class"] = ood_true_names
    err_df["pred_class"] = ood_pred_names
    err_df["pred_confidence"] = probs.max(axis=1)
    err_df = err_df[err_df["true_class"] != err_df["pred_class"]].sort_values("pred_confidence", ascending=False)
    err_df[["sample_id", "country", "region", "distance_to_country_boundary_km", "true_class", "pred_class", "pred_confidence"]].head(50).to_csv(
        OUT_DIR / "m5_error_analysis_samples.csv", index=False
    )

    # Plot Figure 2: Side-by-side normalized confusion matrices for Dev Val vs OOD Test
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (title, yt, yp) in zip(
        axes,
        [("ResNet-18 Dev Validation", val_true_names, val_pred_names), ("ResNet-18 Geographic OOD Test", ood_true_names, ood_pred_names)],
    ):
        cm = confusion_matrix(yt, yp, labels=classes, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "m5_confusion_matrices.png", dpi=300)
    plt.close()

    # 4. SENSITIVITY, ROBUSTNESS & ABLATION

    print("\n--- Step 4: Sensitivity, Robustness & Ablation ---")

    # Sensitivity Check 1: Sweep learning rates from 1e-4 to 3e-3 on validation
    lr_vals = [1e-4, 3e-4, 1e-3, 3e-3]
    lr_scores = []
    for lv in lr_vals:
        r = train_and_eval_resnet(train_df, val_df, ood_df, class_to_idx, seed=42, lr=lv, epochs=10, eval_ood=False)
        log_run("ResNet-18 (Sensitivity LR)", f"lr={lv}, epochs=10", 42, r["val_f1"], None, r["runtime_min"], "LR sensitivity")
        lr_scores.append(r["val_f1"])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([str(x) for x in lr_vals], lr_scores, marker="o", linewidth=2, color="#2B5C8F")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Dev Validation Macro-F1")
    ax.set_title("ResNet-18 Learning Rate Sensitivity")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "m5_fig2_lr_sensitivity.png", dpi=300)
    plt.close()

    # Sensitivity Check 2: Filter out OOD images near country borders (0, 2, 5, 10 km buffers)
    buffer_rows = []
    for buf_km in [0.0, 2.0, 5.0, 10.0]:
        mask = (ood_df["distance_to_country_boundary_km"] >= buf_km).to_numpy()
        bf1 = f1_score(seed42_artifacts["rn_ood_true"][mask], seed42_artifacts["rn_ood_preds"][mask], average="macro", zero_division=0)
        buffer_rows.append({"border_buffer_km": buf_km, "n_ood_samples": int(mask.sum()), "resnet_ood_macro_f1": round(bf1, 4)})
    pd.DataFrame(buffer_rows).to_csv(OUT_DIR / "m5_border_buffer_sensitivity.csv", index=False)

    # Robustness Check: Test both models under Gaussian sensor noise and brightness/contrast shifts
    rob_rows = []
    rn_model = seed42_artifacts["rn_model"]
    rf_pipe = seed42_artifacts["rf_pipe"]

    for mode, label in [(None, "Clean OOD"), ("noise", "Gaussian Noise (sigma=0.08)"), ("brightness", "Brightness/Contrast Shift")]:
        X_ood_corr = X_ood_rf if mode is None else extract_flat_rgb(ood_df, corrupt_mode=mode)
        rf_f1 = f1_score(y_ood, rf_pipe.predict(X_ood_corr), average="macro", zero_division=0)

        class CorruptedDS(Dataset):
            def __getitem__(self, idx):
                with Image.open(ood_df["path"].iloc[idx]) as img:
                    img = img.convert("RGB")
                    if mode == "brightness":
                        img = ImageEnhance.Brightness(img).enhance(1.35)
                        img = ImageEnhance.Contrast(img).enhance(1.25)
                    t = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img)
                    if mode == "noise":
                        torch.manual_seed(idx)
                        t = t + torch.randn_like(t) * 0.15
                return t, class_to_idx[ood_df["class"].iloc[idx]]
            def __len__(self):
                return len(ood_df)

        loader = DataLoader(CorruptedDS(), batch_size=128, shuffle=False)
        rp, rt_arr, _ = predict_resnet(rn_model, loader)
        rn_f1 = f1_score(rt_arr, rp, average="macro", zero_division=0)
        rob_rows.append({"condition": label, "rf_ood_macro_f1": round(rf_f1, 4), "resnet_ood_macro_f1": round(rn_f1, 4)})
    pd.DataFrame(rob_rows).to_csv(OUT_DIR / "m5_robustness.csv", index=False)

    # Ablation Study: Remove one component at a time from our full ResNet-18 setup
    full_ood_f1 = float(per_seed_df[(per_seed_df["model"] == "ResNet-18 (Proposed)") & (per_seed_df["seed"] == 42)]["ood_f1"].iloc[0])
    ablation_variants = [
        ("Full ResNet-18 (64x64, Pretrained, Aug, Weighted)", dict(pretrained=True, augment=True, class_weighted=True, img_size=64), full_ood_f1),
        ("w/o Data Augmentation", dict(pretrained=True, augment=False, class_weighted=True, img_size=64), None),
        ("w/o Class-Weighted Loss", dict(pretrained=True, augment=True, class_weighted=False, img_size=64), None),
        ("w/o ImageNet Pretraining (Random Init)", dict(pretrained=False, augment=True, class_weighted=True, img_size=64), None),
        ("Downsampled Input (16x16 Resolution)", dict(pretrained=True, augment=True, class_weighted=True, img_size=16), None),
    ]
    abl_rows = []
    for v_name, kw, preset_f1 in ablation_variants:
        if preset_f1 is not None:
            of1 = preset_f1
            rid = "E027"
        else:
            r = train_and_eval_resnet(
                train_df, val_df, ood_df, class_to_idx, seed=42,
                lr=best_rn_cfg["lr"], batch_size=best_rn_cfg["batch_size"],
                weight_decay=best_rn_cfg["weight_decay"], epochs=15, eval_ood=True, **kw
            )
            of1 = r["ood_f1"]
            rid = log_run(f"Ablation: {v_name}", str(kw), 42, r["val_f1"], of1, r["runtime_min"], "Ablation study")
        abl_rows.append({
            "run_id": rid,
            "variant": v_name,
            "ood_macro_f1": round(of1, 4),
            "delta_vs_full": round(of1 - full_ood_f1, 4),
        })
    pd.DataFrame(abl_rows).to_csv(OUT_DIR / "m5_ablation.csv", index=False)

    # Export the full 47-run experiment log and hardware/software environment metadata
    pd.DataFrame(exp_log).to_csv(OUT_DIR / "m5_experiment_log.csv", index=False)
    env_meta = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "git_commit": get_git_commit(),
        "seeds": SEEDS,
        "split_seed": SPLIT_SEED,
    }
    with open(OUT_DIR / "m5_environment_meta.json", "w") as f:
        json.dump(env_meta, f, indent=2)

    print("\nALL MILESTONE 5 EXPERIMENTS COMPLETE!")
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
