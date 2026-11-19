# Out-of-Distribution Generalization in Satellite Land Cover Classification

**Course:** CSCI 3052U — Machine Learning I  
**Team:** SixtyFive.AI

## Project Overview

This project investigates how geographic domain shift affects satellite image classification using the EuroSAT Sentinel-2 dataset. We train models exclusively on scenes from Western and Central Europe (**Development**, $N = 17,580$) and evaluate them on a geographically separated **Out-of-Distribution (OOD) Test** set ($N = 6,231$) covering Southern and Eastern Europe.

Our primary evaluation metric is **Macro-F1**, which gives equal weight to each land-cover class despite regional class-frequency shifts.

## Dataset

We use the official EuroSAT v2 dataset, containing 27,000 RGB images across 10 land-cover classes (3,189 patches are excluded during geographic partitioning: 2,074 outside the selected regions and 1,115 unresolved country assignments).

- Official dataset: https://zenodo.org/records/7711810
- RGB images: `data/EuroSAT_RGB/EuroSAT_RGB/`

Raw datasets are excluded from Git due to repository size limits. Note that `src/run_milestone5.py` will also download and extract `EuroSAT_RGB` automatically if it is not already present locally.

## Repository Structure

- `notebooks/` — Exploratory data analysis and dataset inspection.
- `src/` — Scripts for geographic manifest generation, baseline training, Milestone 5 ResNet-18 evaluation, and result aggregation.
- `outputs/` — Generated metadata, geographic manifests, Milestone 5 metric tables, experiment logs, and 300 DPI figures.
- `reports/` — Project proposals, progress memos, and the Milestone 5 experimental results report (`reports/milestone5_results.md`).
- `requirements.txt` — Python dependencies.

## Environment Setup

Use Python 3.11 or another compatible Python version. From the repository root, install the dependencies:

    python -m pip install -r requirements.txt

The project uses NumPy, pandas, Pillow, scikit-learn, SciPy, PyTorch, torchvision, Matplotlib, Rasterio, Shapely, PyProj, and PyShp.

## Data Preparation

1. Download and extract the official EuroSAT v2 RGB archive into `data/EuroSAT_RGB/EuroSAT_RGB/`.
2. Obtain the Natural Earth Admin 0 Countries 1:10m boundary dataset and extract its shapefile components into `data/natural_earth/ne_10m_admin_0_countries/`.
3. Run `notebooks/01_eda_eurosat.ipynb` to inspect the dataset and generate the initial metadata artifacts.
4. From the repository root, build the geographic manifest:

    python src/build_geographic_manifest.py

This generates `outputs/geographic_manifest_rebuilt.csv`, assigning patches to `development` or `ood_test` partitions and computing distances to national boundaries.

## Running the Experiments

### Milestone 5 Full Pipeline (Tuning, 5-Seed Evaluation, Error Analysis, Robustness & Ablations)

Run the end-to-end Milestone 5 pipeline from the repository root:

    python -u src/run_milestone5.py

This script performs:
1. **Validation-Only Hyperparameter Tuning:** 12 random grid trials each for PCA + Random Forest and ResNet-18 (`Run IDs E001–E024`).
2. **5-Seed Evaluation (`seeds = 42, 100, 2026, 777, 999`):** Trains and evaluates the Majority-Class baseline, tuned PCA + Random Forest, and tuned ResNet-18 (`Run IDs E025–E039`), computing 95% confidence intervals, Wilcoxon signed-rank tests, and bootstrap CIs.
3. **Error Analysis & Confusion Matrices:** Exports per-class $F_1$ drops and the top 50 highest-confidence OOD errors.
4. **Sensitivity, Robustness & Ablation Studies:** Evaluates learning-rate sensitivity, $0\text{–}10\text{ km}$ border-buffer sensitivity, Gaussian noise and brightness/contrast perturbations, and component ablations (`Run IDs E040–E047`).

### Earlier Milestone Baselines

    python src/baseline.py
    python src/run_rgb_baseline.py
    python src/combine_results.py

## Milestone 5 Results Summary (5 Seeds)

All models were tuned on Development Validation ($n = 3,516$) and evaluated across 5 random seeds on Geographic OOD Test ($N = 6,231$). Full tables, figures, and the 47-run experiment log are documented in [`reports/milestone5_results.md`](reports/milestone5_results.md).

| Model | Dev Val Accuracy | Dev Val Macro-F1 | Geographic OOD Accuracy | Geographic OOD Macro-F1 |
|---|---:|---:|---:|---:|
| Majority-Class Baseline | 0.1359 ± 0.0000 | 0.0239 ± 0.0000 | 0.0913 ± 0.0000 | 0.0167 ± 0.0000 |
| PCA + Random Forest (400 trees) | 0.7157 ± 0.0028 | 0.6997 ± 0.0027 | 0.4481 ± 0.0022 | 0.4374 ± 0.0024 |
| **ResNet-18 (Proposed, 64×64)** | **0.9902 ± 0.0008** | **0.9903 ± 0.0008** | **0.8809 ± 0.0046** | **0.8561 ± 0.0054** |

### Key Takeaways

- **Substantial OOD Improvement:** Fine-tuned ResNet-18 achieves **0.8561 ± 0.0054** Macro-F1 on unseen Southern and Eastern European regions, nearly doubling the PCA + Random Forest baseline (**0.4374 ± 0.0024**) and cutting the relative geographic generalization drop from **37.5%** down to **13.6%**.
- **Primary Drivers of Generalization:** Ablation studies (`outputs/m5_ablation.csv`) show that preserving native $64 \times 64$ spatial resolution (**+0.2527** OOD Macro-F1 over $16 \times 16$) and ImageNet pretraining (**+0.1479** over random initialization) account for the largest gains.
- **Error Modes & Border Robustness:** OOD errors concentrate in climatically sensitive agricultural and riparian classes (`Pasture`, `PermanentCrop`, `River`) in Mediterranean regions, while border-buffer checks ($0\text{–}10\text{ km}$) confirm OOD accuracy remains stable (**0.8599** to **0.8635**) well inside country interiors.

## Reproducibility and Limitations

- All experiments use fixed seeds (`SPLIT_SEED = 42`, evaluation seeds `[42, 100, 2026, 777, 999]`) and log environment metadata in `outputs/m5_environment_meta.json`.
- Raw datasets and local model checkpoints are excluded from Git via `.gitignore`.
- Severe OOD class scarcity (e.g., `Pasture` having $n = 54$ OOD samples) and sensitivity to additive sensor noise remain key limitations for future work.
