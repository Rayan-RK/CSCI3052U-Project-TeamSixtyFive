# Out-of-Distribution Generalization in Satellite Land Cover Classification

**Course:** CSCI 3052U — Machine Learning I  
**Team:** SixtyFive.AI

## Project Overview

This project investigates how geographic domain shift affects satellite image classification using the EuroSAT Sentinel-2 dataset. We compare model performance on a development validation set and a geographically separated out-of-distribution (OOD) test set.

Our primary evaluation metric is **Macro-F1**, which gives equal weight to each land-cover class.

## Dataset

We use the official EuroSAT v2 dataset, containing 27,000 RGB images and 27,000 multispectral images across 10 land-cover classes.

- Official dataset: https://zenodo.org/records/7711810
- RGB images: `data/EuroSAT_RGB/EuroSAT_RGB/`
- Multispectral images: `data/EuroSAT_MS/EuroSAT_MS/`

Raw datasets are not included in this repository because of their size. Download the official archives and extract them into the corresponding directories above.

## Repository Structure

- `notebooks/` — Exploratory data analysis and dataset inspection.
- `src/` — Scripts for geographic manifest generation, baseline training, and result aggregation.
- `outputs/` — Generated metadata, geographic manifests, evaluation results, and figures.
- `reports/` — Project proposals, team documentation, and milestone reports.
- `requirements.txt` — Python dependencies.

## Environment Setup

Use Python 3.11 or another compatible Python version. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The project uses Python libraries including NumPy, pandas, Pillow, scikit-learn, Rasterio, Shapely, PyProj, and PyShp.

## Data Preparation

1. Download and extract the official EuroSAT v2 RGB and multispectral archives into the directories specified above.
2. Obtain the Natural Earth Admin 0 Countries 1:10m boundary dataset and extract its shapefile components into:

   `data/natural_earth/ne_10m_admin_0_countries/`

3. Run all cells in `notebooks/01_eda_eurosat.ipynb` to inspect the dataset and generate the metadata artifacts required by the geographic pipeline.
4. From the repository root, build the geographic manifest:

```bash
python src/build_geographic_manifest.py
```

This generates `outputs/geographic_manifest_rebuilt.csv`. The script uses sample coordinates and country boundaries to assign candidate partitions. Samples within 1 km of a country boundary are flagged for review, and unmatched samples remain unresolved.

The geographic assignments are provisional and should be validated before final scientific conclusions are drawn.

## Baseline Experiments

Run the majority-class baseline:

```bash
python src/baseline.py
```

Run the RGB PCA + Random Forest baseline:

```bash
python src/run_rgb_baseline.py
```

Combine the summary results:

```bash
python src/combine_results.py
```

The RGB baseline resizes images to 16 × 16 pixels, applies PCA with 100 components, and trains a Random Forest with 200 trees. The development data is divided into training and validation subsets using an 80/20 stratified split and random seed 42.

## Current Preliminary Results

| Model | Partition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Majority-class baseline | Development validation | 0.1360 | 0.0239 |
| Majority-class baseline | OOD test | 0.0913 | 0.0167 |
| PCA + Random Forest (RGB) | Development validation | 0.7085 | 0.6910 |
| PCA + Random Forest (RGB) | OOD test | 0.4486 | 0.4365 |

The RGB baseline shows a preliminary Macro-F1 decrease of approximately 0.255 between development validation and OOD test. Per-class results are available in `outputs/rgb_per_class_results.csv`, with F1-gap analysis in `outputs/rgb_f1_gap_analysis.csv`.

These results are preliminary. Differences in geographic coverage and class distributions may contribute to the observed performance gap.

## Reproducibility and Limitations

- The experiments use a fixed random seed for the development train/validation split and model initialization.
- The geographic manifest is generated from sample coordinates and Natural Earth country boundaries.
- Boundary screening, unresolved samples, and class imbalance may affect the evaluation.
- Raw datasets and other large files are excluded from Git.
- Results should be regenerated after changes to the dataset, geographic assignments, dependencies, or model configuration.

