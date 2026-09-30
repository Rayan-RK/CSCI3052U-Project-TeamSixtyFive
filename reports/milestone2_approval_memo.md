# Out-of-Distribution Generalization in Satellite Land Cover Classification: A Geographic Domain-Shift Approach
**Milestone 2 Approval Documentation & Technical Plan**  
**CSCI 3052U — Machine Learning I  |  Team SixtyFive.AI**

## 1. Data Source & License
* **Dataset:** EuroSAT (Sentinel-2 multispectral imagery dataset).
* **Source & Provenance:** Collected from Sentinel-2 satellite imagery spanning 34 European countries (Helber et al., 2019).
* **Licensing:** Released under the permissive MIT License, fully satisfying open-access and legal reproducibility guidelines.

## 2. Comprehensive Data Card
* **Total Instances:** 26,998 geo-referenced image patches.
* **Classes (10 Categories):** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, and Sea/Lake.
* **Feature Representation:** Fixed spatial resolution of $64 \times 64$ pixels with a 10-meter Ground Sampling Distance (GSD).
* **Data Quality & Missingness:** Fully complete dataset with zero missing values or corrupt files. Class distribution is balanced, containing roughly 2,000 to 3,000 images per category.

## 3. Sample Inputs and Outputs
* **Input ($X$):** A $64 \times 64$ multi-channel raster patch extracted from orbital imagery representing a specific geographic coordinate block.
* **Output ($Y$):** A discrete multi-class integer label corresponding to one of the 10 land cover classes.

## 4. Initial Exploratory Data Analysis (EDA) Plan
An EDA notebook has been initialized in the repository (`notebooks/01_eda_eurosat.ipynb`) to execute the following checks:
* Verify class frequency counts to ensure no severe imbalance skew.
* Render sample image tensors across all 10 classes to inspect visual variance.
* Compute channel-wise pixel mean and standard deviation arrays for data normalization.

## 5. Target / Task Definition
* **Supervised Task:** Multi-class image classification.
* **Core Objective:** Predict the precise land use/land cover category from unseen spatial zones to evaluate model robustness against out-of-distribution geographic domain shifts.

## 6. Risk Assessment & Mitigation
| Risk & Description | Mitigation Strategy |
| :--- | :--- |
| **Spatial Autocorrelation Leakage:** Standard random splits mix patches from identical regional grids into both training and test subsets. | Enforce a strict geographic metadata-based split. |
| **Overfitting to Local Textures:** CNNs may latch onto country-specific road layouts or building styles. | Use cross-validation restricted strictly to training zones and evaluate domain degradation metrics. |

## 7. Train / Validation / Test Strategy (OOD Spatial Partitioning)
To explicitly test Out-of-Distribution Generalization, we reject standard random partitioning:
* **Training / Validation Set (75%):** Image patches sourced from Western and Central European coordinate tiles. Hyperparameter tuning occurs exclusively here.
* **Out-of-Distribution Test Set (25%):** Spatially disjoint image patches sourced from distinct Southern and Eastern European coordinate boundaries to evaluate real-world transfer degradation.

## 8. Primary Metric
* **Macro-$F_1$ Score:** Selected over overall accuracy to account for subtle class variations and ensure balanced performance across all 10 land cover categories. We target measuring a precise degradation delta ($\Delta$ Macro-$F_1$) between standard IID cross-validation and spatial OOD evaluation.

## 9. Feasibility & Compute Plan
* **Hardware Footprint:** Dataset size is approximately 2 GB, lightweight enough to run locally on standard hardware or via cloud-accelerated environments (Google Colab / university servers) equipped with standard GPUs.
* **Software Stack:** Python 3.10+, PyTorch (for ResNet-18 fine-tuning), Scikit-Learn (for PCA and Random Forest baselines), NumPy, Pandas, and Rasterio/Albumentations.
