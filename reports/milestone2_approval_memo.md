# Out-of-Distribution Generalization in Satellite Land Cover Classification: A Geographic Domain-Shift Approach
**Milestone 2 Approval Documentation & Technical Plan**  
**CSCI 3052U — Machine Learning I  |  Team SixtyFive.AI**

## 1. Data Source & License
* **Dataset:** EuroSAT (Sentinel-2 multispectral imagery dataset).
* **Source & Provenance:** Collected from Sentinel-2 satellite imagery spanning 34 European countries (Helber et al., 2019)[cite: 6].
* **Licensing:** The [official EuroSAT repository](https://github.com/phelber/EuroSAT) identifies the dataset as MIT-licensed and asks users to observe the Copernicus Sentinel Data Terms and Conditions[cite: 6]. Record the release and retain source attribution[cite: 6].

## 2. Comprehensive Data Card
* **Total Instances:** 27,000 labeled patches, verified in both the RGB and multispectral v2 archives[cite: 6]. These are two representations of the same 27,000 samples, not 54,000 independent observations[cite: 6].
* **Classes (10 Categories):** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, and Sea/Lake[cite: 6].
* **Feature Representation:** Observed arrays are 64 × 64 × 3 for RGB and 64 × 64 × 13 for MS[cite: 6]. GeoTIFF transforms encode an approximately 10 m output pixel grid; this does not imply all Sentinel-2 source bands have native 10 m resolution[cite: 6].
* **Data Quality & Missingness:** All 54,000 discovered image files were decoded (27,000 in each representation): 0 decode failures, 0 unexpected shapes, 0 non-finite stored values, and 0 masked/nodata values were detected[cite: 6]. Both archive MD5 digests match the publisher's checksums; extracted image paths match their archive members with no missing or extra paths[cite: 6]. These findings apply to this release and these checks, not to unobserved scenes, semantic label errors, or all possible quality defects[cite: 6].

## 3. Sample Inputs and Outputs
* **Input ($X$):** A $64 \times 64$ multi-channel raster patch extracted from orbital imagery representing a specific geographic coordinate block[cite: 6].
* **Output ($Y$):** A discrete multi-class integer label corresponding to one of the 10 land cover classes[cite: 6].

## 4. Initial Exploratory Data Analysis (EDA): Measured Evidence
The executed notebook, `notebooks/01_eda_eurosat.ipynb`, and tables/figures in `outputs/eda/` record actual inspection of [EuroSAT v2, Zenodo record 7711810](https://zenodo.org/records/7711810)[cite: 6]. Both complete archives passed the published MD5 checks and extraction CRC checks[cite: 6]. This analysis was run on 2026-10-02 (America/Toronto)[cite: 6].

| Class | RGB files | MS files |
| :--- | ---: | ---: |
| AnnualCrop | 3,000 | 3,000[cite: 6] |
| Forest | 3,000 | 3,000[cite: 6] |
| HerbaceousVegetation | 3,000 | 3,000[cite: 6] |
| Highway | 2,500 | 2,500[cite: 6] |
| Industrial | 2,500 | 2,500[cite: 6] |
| Pasture | 2,000 | 2,000[cite: 6] |
| PermanentCrop | 2,500 | 2,500[cite: 6] |
| Residential | 3,000 | 3,000[cite: 6] |
| River | 2,500 | 2,500[cite: 6] |
| SeaLake | 3,000 | 3,000[cite: 6] |

The largest class has 3,000 samples and the smallest, Pasture, has 2,000 (1.5:1 ratio)[cite: 6]. This is moderate class-frequency variation, not exact balance; Macro-F1 gives each class equal weight[cite: 6]. Counts match across representations and all 27,000 class/sample identifiers have one-to-one RGB/MS correspondence[cite: 6].

All images have the expected shape and decoded successfully[cite: 6]. The notebook saves deterministic representative samples with their labels and filenames for all ten classes[cite: 6]. RGB examples show different land-cover textures; MS previews use bands 4/3/2 with a display-only percentile stretch[cite: 6]. Visual inspection is illustrative, not a comprehensive label audit[cite: 6]. See [RGB examples](../outputs/eda/rgb_samples.png) and [MS examples](../outputs/eda/ms_samples.png)[cite: 6].

Pixel-weighted RGB means on [0,1] are **(0.344376, 0.380291, 0.407770)**, with population standard deviations **(0.202661, 0.136897, 0.115550)** in R/G/B order[cite: 6]. The complete 13-band MS statistics are in `outputs/eda/channel_statistics.csv` in raw stored units[cite: 6]. These whole-dataset summaries are descriptive; model normalization and PCA must be fitted separately on each training fold, excluding validation and OOD samples[cite: 6].

All 27,000 MS files were inspected for CRS, affine transform, bounds and WGS84 centers; **27,000** yielded valid candidate coordinates, with **0** coordinate failures[cite: 6]. Country mapping and provisional regional coverage are described below[cite: 6]. No model was trained and no performance score is claimed at this milestone[cite: 6].

## 5. Target / Task Definition
* **Supervised Task:** Multi-class image classification[cite: 6].
* **Core Objective:** Predict the precise land use/land cover category from unseen spatial zones to evaluate model robustness against out-of-distribution geographic domain shifts[cite: 6].

## 6. Risk Assessment & Mitigation
| Risk & Description | Mitigation Strategy |
| :--- | :--- |
| **Spatial Autocorrelation Leakage:** Standard random splits mix patches from identical regional grids into both training and test subsets[cite: 6]. | Enforce a strict geographic metadata-based split[cite: 6]. |
| **Overfitting to Local Textures:** CNNs may latch onto country-specific road layouts or building styles[cite: 6]. | Use cross-validation restricted strictly to training zones and evaluate domain degradation metrics[cite: 6]. |

## 7. Train / Validation / Test Strategy (OOD Spatial Partitioning)
The following strategy preserves the Research Proposal's geographic-domain experiment[cite: 6]. Coordinate availability has now been verified; the final leakage-audited split remains to be frozen before modelling[cite: 6]:

* **Geographic metadata prerequisite:** The notebook extracts each patch's coordinate reference system (CRS), affine transform, bounds, and center coordinates from the multispectral GeoTIFF metadata; it transforms centers into WGS84 longitude/latitude[cite: 6]. RGB JPEG filenames and class folders do not establish geographic location[cite: 6]. If RGB inputs are used, match them to GeoTIFFs by class and sample identifier and verify one-to-one coverage[cite: 6].
* **Region mapping and initial feasibility:** All candidate centers were spatially joined to [Natural Earth Admin 0 Countries 1:10m v5.1.1](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/)[cite: 6]. The notebook and `outputs/eda/provenance.json` provide the explicit, mutually exclusive study-defined country-to-region mapping[cite: 6]. A patch enters a candidate partition only when its center matches one country and its transformed bounding box is covered by that polygon[cite: 6]. Other regions are excluded and border/coast/unmatched cases remain flagged[cite: 6]. Preliminary counts: **17,847 development**, **6,334 OOD**, **2,213 outside study regions**, and **606 unresolved**[cite: 6]. Among the 24,181 candidates, the realized proportions are **73.8%/26.2%**, not a verified 75%/25% split[cite: 6]. Per-class coverage is retained in the notebook and sample-level assignments in `outputs/eda/preliminary_region_manifest.csv`[cite: 6].
* **Geographic group audit before modelling:** Generalized country polygons and center coordinates alone do not establish leakage-free groups[cite: 6]. Define groups using source tiles where available or documented spatial blocks; inspect neighboring/overlapping patches and apply a documented buffer/exclusion rule across folds and the OOD boundary[cite: 6]. Freeze a versioned sample/group/fold manifest after this audit[cite: 6]. Do not use preliminary candidate partitions as final training splits without these checks[cite: 6].
* **Training / Validation Set (target 75%):** Western and Central European samples form the development partition[cite: 6]. Hyperparameter tuning will use geographic-grouped cross-validation within this partition (planned five folds, subject to sufficient groups and class coverage)[cite: 6]. Each fold holds out entire geographic groups for validation; geographically related samples must remain together[cite: 6]. Record fold assignments and class counts, assert disjoint group membership, and fit all learned preprocessing on the training fold only[cite: 6]. Select settings by mean validation Macro-$F_1$, then refit on the full development partition[cite: 6].
* **Out-of-Distribution Test Set (target 25%):** Southern and Eastern European samples form the spatially disjoint held-out test partition[cite: 6]. Keep it excluded from tuning, learned preprocessing, and model selection; evaluate after the development procedure is fixed[cite: 6].
* **Feasibility and comparison:** The 75%/25% proportions are targets, not measured counts or a reason to move individual patches across geographic boundaries[cite: 6]. Report the realized proportions and per-class coverage after mapping[cite: 6]. If the metadata or regional coverage is insufficient, document the blocker and revise the protocol before claiming a geographic OOD result[cite: 6]. Preserve the proposal's separate stratified-random/IID comparison within the development data; it does not replace geographic-grouped tuning or access the held-out OOD set[cite: 6].

## 8. Primary Metric
* **Macro-$F_1$ Score:** Selected over overall accuracy to account for subtle class variations and ensure balanced performance across all 10 land cover categories[cite: 6]. We target measuring a precise degradation delta ($\Delta$ Macro-$F_1$) between standard IID cross-validation and spatial OOD evaluation[cite: 6].

## 9. Feasibility & Compute Plan
* **Hardware Footprint:** The verified downloads are 94,658,721 bytes (RGB) and 2,065,402,329 bytes (MS)[cite: 6]. Allow approximately 6 GB for archives plus extracted data, with additional space for outputs and environments[cite: 6]. Initial EDA runs on CPU; later ResNet-18 experiments are planned for a GPU-equipped local or cloud environment, subject to team access[cite: 6].
* **Software Stack:** Python 3.10+, PyTorch (for ResNet-18 fine-tuning), Scikit-Learn (for PCA and Random Forest baselines), NumPy, Pandas, and Rasterio/Albumentations[cite: 6].

## 10. Milestone 2 Evidence and Remaining Decisions
The notebook contains executed counts, sample images, file/shape checks, pixel statistics, archive verification and geographic metadata/coverage[cite: 6]. `outputs/eda/provenance.json` records versions, checksums and the region convention[cite: 6]. The companion research proposal has been updated for these findings[cite: 6]. Remaining pre-training decisions are the final geographic grouping/buffer policy and review of excluded/ambiguous patches; these are limitations of the planned experiment, not completed model results[cite: 6].

**Tool assistance and verification:** Codex assisted with the code and prose[cite: 6]. Code was checked with synthetic failure cases and then executed on the checksum-verified official data; reported numbers come from saved outputs[cite: 6]. The team must independently review and understand the notebook, source claims, and submission before submitting it[cite: 6].
