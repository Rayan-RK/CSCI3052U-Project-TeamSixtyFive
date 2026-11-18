# Milestone 5: Experimental Results & Analysis
**Project:** Geographic Out-of-Distribution Land-Cover Classification on EuroSAT  
**Team:** Team Sixty-Five (CSCI 3052U)  
**Repository Commit:** `0668fe9`  

---

## 4. Experimental Results

### 4.1 Evaluation Setup and Baselines
To test if a model trained on satellite imagery from one part of Europe still works on unseen regions without spatial leakage, we held out all Southern and Eastern European images as our **Geographic OOD Test** set ($N = 6,231$). All training and model selection was done strictly on the **Development** set ($N = 17,580$ images from Northern, Western, and Central Europe). We used an 80/20 stratified split (`seed = 42`) to divide the Development set into **Development Train** ($n = 14,064$) and **Development Validation** ($n = 3,516$), and dropped the 3,189 patches that had `NoData` coordinates before splitting. Since class frequencies change a lot across regions—for example, `Pasture` only has $n = 54$ images in the OOD Test set while `HerbaceousVegetation` has $n = 1,414$—we report **Macro-$F_1$** as our main metric along with overall accuracy.

We tested three models on these exact same splits:
1. **Majority-Class Baseline:** Always predicts the most common class in the training set (`DummyClassifier(strategy="most_frequent")`) to give us a floor for comparison.
2. **PCA + Random Forest:** Resizes each image to $16 \times 16$, flattens the RGB values into a 768-feature vector, runs 100-component PCA, and fits a class-balanced Random Forest.
3. **ResNet-18 (Our Proposed Model):** Takes a pretrained ResNet-18 and fine-tunes it on the full $64 \times 64$ images using AdamW, cosine annealing learning rate decay, random flips and $\pm 15^\circ$ rotations, and class-weighted cross-entropy loss to handle class imbalance.

---

### 4.2 Hyperparameter Tuning
To keep the comparison fair and avoid tuning on the test set, we gave both trainable models an equal budget of **12 random grid trials** evaluated only on **Development Validation** (`Run IDs E001–E024`). For PCA + Random Forest, we searched `n_estimators` $\in \{100, 200, 400\}$, `max_depth` $\in \{15, 25, \text{None}\}$, and `max_features` $\in \{\text{sqrt}, \text{log2}\}$. For ResNet-18, we searched learning rate $\text{lr} \in \{1\times 10^{-4}, 3\times 10^{-4}, 1\times 10^{-3}, 3\times 10^{-3}\}$, batch size $\in \{32, 64\}$, and weight decay $\in \{1\times 10^{-5}, 1\times 10^{-4}\}$ for 10 epochs per trial.

**Table 1: Validation Hyperparameter Search Summary (12 Trials per Model)**

| Model Family | Parameters Searched (12 Trials Each) | Best Config Found | Run ID | Dev Val Macro-$F_1$ |
| :--- | :--- | :--- | :---: | :---: |
| **PCA + Random Forest** | `n_estimators`, `max_depth`, `max_features` (`pca=100`) | `n_estimators=400`, `max_depth=None`, `max_features='sqrt'` | `E012` | **0.7024** |
| **ResNet-18 (Proposed)** | `lr`, `batch_size`, `weight_decay` (10 epochs) | `lr=3e-4`, `batch_size=64`, `weight_decay=1e-5` (trained 15 epochs for final runs) | `E023` | **0.9893** |

---

### 4.3 Multi-Seed Performance Comparison
After picking the best hyperparameters from validation, we trained each model across **5 random seeds (`42, 100, 2026, 777, 999`)** (`Run IDs E025–E039`). Table 2 shows the mean, standard deviation ($\text{SD}$), and 95% Student-$t$ confidence intervals ($\text{df} = 4$) across those 5 runs, and **Figure 1** (`outputs/m5_fig1_model_comparison_ci.png`) plots the Macro-$F_1$ comparison.

**Table 2: Results Across 5 Seeds on Development Validation and Geographic OOD Test**

| Model | Dev Val Accuracy (Mean $\pm$ SD [95% CI]) | Dev Val Macro-$F_1$ (Mean $\pm$ SD [95% CI]) | Geographic OOD Accuracy (Mean $\pm$ SD [95% CI]) | Geographic OOD Macro-$F_1$ (Mean $\pm$ SD [95% CI]) | Avg. Train Time (min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | $0.1359 \pm 0.0000$ $[0.1359, 0.1359]$ | $0.0239 \pm 0.0000$ $[0.0239, 0.0239]$ | $0.0913 \pm 0.0000$ $[0.0913, 0.0913]$ | $0.0167 \pm 0.0000$ $[0.0167, 0.0167]$ | 0.00 |
| **PCA + Random Forest** | $0.7157 \pm 0.0028$ $[0.7122, 0.7192]$ | $0.6997 \pm 0.0027$ $[0.6963, 0.7030]$ | $0.4481 \pm 0.0022$ $[0.4453, 0.4509]$ | $0.4374 \pm 0.0024$ $[0.4343, 0.4404]$ | 0.12 |
| **ResNet-18 (Proposed)** | **$0.9902 \pm 0.0008$** $[0.9892, 0.9911]$ | **$0.9903 \pm 0.0008$** $[0.9893, 0.9913]$ | **$0.8809 \pm 0.0046$** $[0.8752, 0.8867]$ | **$0.8561 \pm 0.0054$** $[0.8494, 0.8629]$ | 3.64 |

A couple of clear findings come out of Table 2:
- **ResNet-18 nearly doubles the Random Forest OOD score:** On the held-out Southern and Eastern European test set, ResNet-18 gets **$0.8561 \pm 0.0054$** Macro-$F_1$, which is **+0.4187** higher than PCA + Random Forest ($0.4374 \pm 0.0024$) and **+0.8394** above the Majority-Class baseline.
- **Smaller generalization drop:** When moving from Development Validation to Geographic OOD Test, PCA + Random Forest drops by **$-0.2623$** Macro-$F_1$ (a $37.5\%$ relative loss from $0.6997$). ResNet-18 only drops by **$-0.1342$** ($13.6\%$ relative) and finishes training in about $3.64\text{ min}$ per run on our RTX 2060 SUPER.

---

### 4.4 Uncertainty and Statistical Significance
To verify that these improvements were consistent across seeds and test samples (`outputs/m5_uncertainty_stats.json`), we ran three checks:
- **Consistent scores across seeds:** Across all 5 seeds, ResNet-18's OOD Macro-$F_1$ stays between $0.8506$ and $0.8633$ ($\text{SD} = 0.0054$), giving a tight 95% confidence interval of **$[0.8494, 0.8629]$**.
- **Wilcoxon Signed-Rank Test:** ResNet-18 beats both PCA + Random Forest and the Majority-Class baseline on all 5 seeds ($5/5$ runs), giving $W = 15$ and $p = 0.0625$ (the lowest two-sided exact $p$-value possible for $n = 5$ paired samples).
- **Bootstrap CI on OOD Test:** Resampling the $N = 6,231$ OOD test predictions 1,000 times with replacement (`seed = 42`, `Run ID E027`, point score $0.8599$) gives a 95% bootstrap confidence interval of **$[0.8470, 0.8724]$**.

---

### 4.5 Per-Class Breakdown and Error Analysis
Looking at the per-class scores for `seed = 42` ResNet-18 (`outputs/m5_per_class_f1.csv` and the confusion matrices in **Figure 2**, `outputs/m5_confusion_matrices.png`) shows that the OOD performance drop is concentrated in a few specific land-cover types rather than affecting every class equally.

**Table 3: Per-Class $F_1$ Scores for ResNet-18 (`seed = 42`, `Run ID E027`, Sorted by OOD Drop)**

| Land-Cover Class | Dev Val $F_1$ | Geographic OOD $F_1$ | $\Delta F_1$ (OOD $-$ Val) | OOD Test Support ($n$) |
| :--- | :---: | :---: | :---: | :---: |
| **Pasture** | 0.9838 | 0.5263 | $-0.4575$ | 54 |
| **PermanentCrop** | 0.9892 | 0.7716 | $-0.2176$ | 1,083 |
| **River** | 0.9839 | 0.8152 | $-0.1687$ | 205 |
| **HerbaceousVegetation** | 0.9912 | 0.8452 | $-0.1460$ | 1,414 |
| **AnnualCrop** | 0.9885 | 0.9014 | $-0.0871$ | 569 |
| **Industrial** | 0.9927 | 0.9085 | $-0.0842$ | 564 |
| **Forest** | 0.9988 | 0.9465 | $-0.0523$ | 770 |
| **Highway** | 0.9850 | 0.9446 | $-0.0405$ | 492 |
| **Residential** | 0.9934 | 0.9585 | $-0.0348$ | 682 |
| **SeaLake** | 0.9981 | 0.9811 | $-0.0170$ | 398 |

We pulled the 50 highest-confidence mistakes on the OOD test set (`outputs/m5_error_analysis_samples.csv`, all predicted with $> 0.997$ softmax confidence) to see what the model was getting wrong and where those images came from:
1. **Mediterranean scrubland vs. orchards (`HerbaceousVegetation` $\leftrightarrow$ `PermanentCrop` / `AnnualCrop`):** Over 60% of the worst high-confidence mistakes happen deep inside **Spain**, **Greece**, **Italy**, and **Cyprus**—often more than $60\text{ km}$ from any national border (e.g., `HerbaceousVegetation_845` in Spain at $109.3\text{ km}$ from the border, predicted as `PermanentCrop` with $0.99995$ confidence). Drier Mediterranean vegetation and spaced olive or citrus orchards show bare soil between trees, which looks very similar to row crops in Northern and Central Europe.
2. **Narrow or dry rivers (`River` $\rightarrow$ `AnnualCrop`):** High-confidence river errors in Spain (`River_2188`, confidence $0.99996$) and Romania (`River_411`, confidence $0.99972$) happen when narrow seasonal rivers or dry gravel beds blend into surrounding farm fields at Sentinel-2's 10 m resolution.
3. **Pasture domain shift and low OOD count (`Pasture` $\rightarrow$ `HerbaceousVegetation`):** `Pasture` has the biggest drop ($-0.4575$, OOD $F_1 = 0.5263$). Besides only having 54 OOD test samples, drier grazing land in Southern and Eastern Europe (like `Pasture_1840` in Bulgaria, confidence $0.99972$) doesn't have the uniform green grass seen in Northern/Western European pastures, so the model confuses it with wild `HerbaceousVegetation`. Meanwhile, classes with clear shapes and textures—`SeaLake` ($0.9811$), `Residential` ($0.9585$), `Forest` ($0.9465$), and `Highway` ($0.9446$)—transfer across regions with almost no drop.

---

### 4.6 Sensitivity, Robustness, and Ablation Studies

#### **Hyperparameter and Border-Buffer Sensitivity**
- **Learning Rate Sensitivity (`Figure 3`, `outputs/m5_fig2_lr_sensitivity.png`, `Run IDs E040–E043`):** In our 10-epoch validation sweeps on `seed = 42`, ResNet-18 stays steady across $\text{lr} = 1\times 10^{-4}$ (**0.9850**), $3\times 10^{-4}$ (**0.9854**), and $1\times 10^{-3}$ (**0.9857**), only dropping slightly at $\text{lr} = 3\times 10^{-3}$ (**0.9722**).
- **Border-Buffer Distance Sensitivity (`outputs/m5_border_buffer_sensitivity.csv`):** We wanted to make sure our OOD accuracy wasn't just coming from test patches sitting right across the border from Development countries. When we re-scored `seed = 42` ResNet-18 after filtering out OOD images within $0, 2, 5,$ and $10\text{ km}$ of any country border, Macro-$F_1$ barely changed: **0.8599** ($0\text{ km}$, $n = 6,231$), **0.8596** ($2\text{ km}$, $n = 6,041$), **0.8596** ($5\text{ km}$, $n = 5,435$), and **0.8635** ($10\text{ km}$, $n = 4,501$). That confirms the model generalizes well into the interior of unseen countries.

#### **Perturbation Robustness (`outputs/m5_robustness.csv`)**
To simulate sensor noise and lighting differences across latitudes, we evaluated both `seed = 42` models on modified versions of the OOD test images:
- **Clean OOD:** PCA + Random Forest = **0.4349** | ResNet-18 = **0.8599**
- **Gaussian Noise ($\sigma = 0.08$):** PCA + Random Forest = **0.2979** ($-0.1370$) | ResNet-18 = **0.4885** ($-0.3714$)
- **Brightness/Contrast Shift ($+35\%$ brightness, $+25\%$ contrast):** PCA + Random Forest = **0.1390** ($-0.2959$) | ResNet-18 = **0.6661** ($-0.1938$)

Under strong brightness and contrast shifts, PCA + Random Forest falls apart ($0.1390$), whereas ResNet-18 still holds **0.6661** Macro-$F_1$. Both models take a big hit from additive pixel noise, which suggests adding random noise augmentation during training would help in future work.

#### **Ablation Study (`outputs/m5_ablation.csv`)**
To see which parts of our ResNet-18 setup actually matter for OOD generalization (`seed = 42`), we removed four pieces one at a time while keeping everything else fixed:

**Table 4: Component Ablation Results on Geographic OOD Test (`seed = 42`)**

| Run ID | Model Variant | Dev Val Macro-$F_1$ | Geographic OOD Macro-$F_1$ | $\Delta$ OOD vs. Full |
| :---: | :--- | :---: | :---: | :---: |
| `E027` | **Full ResNet-18 ($64 \times 64$, Pretrained, Aug, Weighted)** | **0.9905** | **0.8599** | — |
| `E044` | w/o Data Augmentation (no flips/rotations) | 0.9870 | 0.8586 | $-0.0013$ |
| `E045` | w/o Class-Weighted Loss (standard cross-entropy) | 0.9887 | 0.8533 | $-0.0066$ |
| `E046` | w/o ImageNet Pretraining (trained from random init) | 0.9598 | 0.7120 | **$-0.1479$** |
| `E047` | Downsampled Input ($16 \times 16$ spatial resolution) | 0.8988 | 0.6072 | **$-0.2527$** |

Two things make the biggest difference in Table 4:
1. **Keeping the full $64 \times 64$ resolution ($-0.2527$ when downsampled to $16 \times 16$):** Shrinking the images down to the $16 \times 16$ size used by the Random Forest drops OOD Macro-$F_1$ from $0.8599$ to $0.6072$. Fine spatial textures—like road edges, building layouts, and crop rows—matter much more across regions than blurry color averages.
2. **ImageNet pretraining ($-0.1479$ without it):** Training ResNet-18 from scratch (`E046`) still gets $0.9598$ on validation, but drops to $0.7120$ on OOD Test. Starting from pretrained weights keeps the network from overfitting to Northern and Central European color palettes.
3. **Class weighting and augmentations ($-0.0066$ and $-0.0013$):** Class-weighted loss helps protect recall on rarer OOD classes, and random flips/rotations give a small extra boost.

---

## 5. Preliminary Conclusions and Limitations

### 5.1 Preliminary Conclusions
Our Milestone 5 experiments show that end-to-end convolutional representations transfer across European geographic regions far better than flattened color-and-PCA baselines. On the held-out Southern and Eastern European OOD test set ( = 6,231$), our fine-tuned ResNet-18 reaches a Macro-$ of **.8561 \pm 0.0054$** across 5 seeds—nearly doubling PCA + Random Forest (.4374 \pm 0.0024$) and cutting the relative generalization drop from Development Validation to Geographic OOD from **37.5%** down to **13.6%**.

Our ablation and sensitivity studies pinpoint why this works:
* **Fine spatial texture matters most:** Downsampling ResNet-18's inputs from  \times 64$ to  \times 16$ (E047) drops OOD Macro-$ by **$-0.2527$**, showing that preserving 10 m structural details (road geometry, building footprints, crop row spacing) is critical when regional vegetation colors shift.
* **Transfer learning prevents regional overfitting:** Removing ImageNet pretraining (E046) drops OOD Macro-$ by **$-0.1479$** even though validation Macro-$ stays high (.9598$).
* **True interior generalization:** Re-evaluating after removing OOD images within , 5,$ and \text{ km}$ of national borders leaves Macro-$ virtually unchanged (.8599 \rightarrow 0.8635$), confirming that performance is not inflated by border-adjacent spatial leakage.

### 5.2 Limitations and Future Work
1. **Severe OOD Class Scarcity (Pasture):** Because EuroSAT was not sampled uniformly across countries, Pasture has only  = 54$ examples in the Southern/Eastern European OOD split and suffers a **$-0.4575$** $ drop (.5263$ OOD $). Expanding evaluation to larger multi-country benchmarks (such as BigEarthNet) would stabilize per-class estimates.
2. **Climate-Driven Spectral Ambiguity:** High-confidence errors cluster in Mediterranean countries (Spain, Greece, Italy, Cyprus), where dry scrubland (HerbaceousVegetation), spaced orchards (PermanentCrop), and seasonal dry riverbeds (River) share exposed soil reflectance. Incorporating Sentinel-2 multispectral bands (e.g., Red-Edge, SWIR, and NDVI indices) alongside RGB should help separate vegetation moisture and crop types.
3. **Sensitivity to Sensor Noise and Calibration:** While ResNet-18 handles global brightness/contrast shifts reasonably well (.6661$ OOD Macro-$), additive Gaussian pixel noise ($\sigma = 0.08$) drops OOD Macro-$ to .4885$. Adding radiometric noise and color-jitter augmentations during training is a straightforward next step.
4. **Excluded NoData Patches:** The 3,189 EuroSAT patches lacking valid coordinate metadata were excluded to keep the geographic boundary split strictly leak-free, reducing total usable sample count by .8\%$.

---

## 5. Preliminary Conclusions and Limitations

### 5.1 Preliminary Conclusions
Our Milestone 5 experiments show that end-to-end convolutional representations transfer across European geographic regions far better than flattened color-and-PCA baselines. On the held-out Southern and Eastern European OOD test set ( = 6,231$), our fine-tuned ResNet-18 reaches a Macro-$ of **.8561 \pm 0.0054$** across 5 seeds—nearly doubling PCA + Random Forest (.4374 \pm 0.0024$) and cutting the relative generalization drop from Development Validation to Geographic OOD from **37.5%** down to **13.6%**.

Our ablation and sensitivity studies pinpoint why this works:
* **Fine spatial texture matters most:** Downsampling ResNet-18's inputs from  \times 64$ to  \times 16$ (E047) drops OOD Macro-$ by **$-0.2527$**, showing that preserving 10 m structural details (road geometry, building footprints, crop row spacing) is critical when regional vegetation colors shift.
* **Transfer learning prevents regional overfitting:** Removing ImageNet pretraining (E046) drops OOD Macro-$ by **$-0.1479$** even though validation Macro-$ stays high (.9598$).
* **True interior generalization:** Re-evaluating after removing OOD images within , 5,$ and \text{ km}$ of national borders leaves Macro-$ virtually unchanged (.8599 \rightarrow 0.8635$), confirming that performance is not inflated by border-adjacent spatial leakage.

### 5.2 Limitations and Future Work
1. **Severe OOD Class Scarcity (Pasture):** Because EuroSAT was not sampled uniformly across countries, Pasture has only  = 54$ examples in the Southern/Eastern European OOD split and suffers a **$-0.4575$** $ drop (.5263$ OOD $). Expanding evaluation to larger multi-country benchmarks (such as BigEarthNet) would stabilize per-class estimates.
2. **Climate-Driven Spectral Ambiguity:** High-confidence errors cluster in Mediterranean countries (Spain, Greece, Italy, Cyprus), where dry scrubland (HerbaceousVegetation), spaced orchards (PermanentCrop), and seasonal dry riverbeds (River) share exposed soil reflectance. Incorporating Sentinel-2 multispectral bands (e.g., Red-Edge, SWIR, and NDVI indices) alongside RGB should help separate vegetation moisture and crop types.
3. **Sensitivity to Sensor Noise and Calibration:** While ResNet-18 handles global brightness/contrast shifts reasonably well (.6661$ OOD Macro-$), additive Gaussian pixel noise ($\sigma = 0.08$) drops OOD Macro-$ to .4885$. Adding radiometric noise and color-jitter augmentations during training is a straightforward next step.
4. **Excluded NoData Patches:** The 3,189 EuroSAT patches lacking valid coordinate metadata were excluded to keep the geographic boundary split strictly leak-free, reducing total usable sample count by .8\%$.

---

## **Part B: Experiment Log**

All 47 runs were executed on **2026-11-17** on commit `0668fe9` using **Python 3.11.9**, **PyTorch 2.6.0+cu124**, and an **NVIDIA GeForce RTX 2060 SUPER** (`outputs/m5_environment_meta.json`). Every run used the fixed `Dev 80/20 (seed 42)` vs. `Geographic OOD` split from `outputs/geographic_manifest_rebuilt.csv`. The full CSV log is saved at `outputs/m5_experiment_log.csv`.

| Run ID | Model / Variant | Hyperparameters | Seed | Val Macro-$F_1$ | OOD Macro-$F_1$ | Runtime (min) | Notes |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `E001` | PCA + Random Forest | `pca=100, n_trees=100, max_depth=25, max_feat=log2` | 42 | 0.6476 | — | 0.03 | RF tuning trial 1/12 |
| `E002` | PCA + Random Forest | `pca=100, n_trees=100, max_depth=15, max_feat=sqrt` | 42 | 0.6770 | — | 0.03 | RF tuning trial 2/12 |
| `E003` | PCA + Random Forest | `pca=100, n_trees=200, max_depth=25, max_feat=sqrt` | 42 | 0.6866 | — | 0.06 | RF tuning trial 3/12 |
| `E004` | PCA + Random Forest | `pca=100, n_trees=400, max_depth=None, max_feat=log2` | 42 | 0.6813 | — | 0.08 | RF tuning trial 4/12 |
| `E005` | PCA + Random Forest | `pca=100, n_trees=400, max_depth=25, max_feat=sqrt` | 42 | 0.6961 | — | 0.12 | RF tuning trial 5/12 |
| `E006` | PCA + Random Forest | `pca=100, n_trees=100, max_depth=25, max_feat=sqrt` | 42 | 0.6798 | — | 0.04 | RF tuning trial 6/12 |
| `E007` | PCA + Random Forest | `pca=100, n_trees=200, max_depth=None, max_feat=log2` | 42 | 0.6758 | — | 0.05 | RF tuning trial 7/12 |
| `E008` | PCA + Random Forest | `pca=100, n_trees=100, max_depth=15, max_feat=log2` | 42 | 0.6468 | — | 0.02 | RF tuning trial 8/12 |
| `E009` | PCA + Random Forest | `pca=100, n_trees=400, max_depth=25, max_feat=log2` | 42 | 0.6706 | — | 0.08 | RF tuning trial 9/12 |
| `E010` | PCA + Random Forest | `pca=100, n_trees=200, max_depth=None, max_feat=sqrt` | 42 | 0.6923 | — | 0.06 | RF tuning trial 10/12 |
| `E011` | PCA + Random Forest | `pca=100, n_trees=200, max_depth=15, max_feat=sqrt` | 42 | 0.6910 | — | 0.06 | RF tuning trial 11/12 |
| `E012` | PCA + Random Forest | `pca=100, n_trees=400, max_depth=None, max_feat=sqrt` | 42 | **0.7024** | — | 0.12 | Selected RF config (trial 12/12) |
| `E013` | ResNet-18 (Proposed) | `lr=0.0001, bs=32, wd=1e-05, epochs=10` | 42 | 0.9854 | — | 3.33 | ResNet-18 tuning trial 1/12 |
| `E014` | ResNet-18 (Proposed) | `lr=0.0001, bs=32, wd=0.0001, epochs=10` | 42 | 0.9838 | — | 2.90 | ResNet-18 tuning trial 2/12 |
| `E015` | ResNet-18 (Proposed) | `lr=0.0001, bs=64, wd=0.0001, epochs=10` | 42 | 0.9843 | — | 2.40 | ResNet-18 tuning trial 3/12 |
| `E016` | ResNet-18 (Proposed) | `lr=0.003, bs=32, wd=0.0001, epochs=10` | 42 | 0.9762 | — | 2.86 | ResNet-18 tuning trial 4/12 |
| `E017` | ResNet-18 (Proposed) | `lr=0.001, bs=32, wd=1e-05, epochs=10` | 42 | 0.9814 | — | 2.90 | ResNet-18 tuning trial 5/12 |
| `E018` | ResNet-18 (Proposed) | `lr=0.001, bs=32, wd=0.0001, epochs=10` | 42 | 0.9829 | — | 3.04 | ResNet-18 tuning trial 6/12 |
| `E019` | ResNet-18 (Proposed) | `lr=0.003, bs=64, wd=0.0001, epochs=10` | 42 | 0.9765 | — | 2.44 | ResNet-18 tuning trial 7/12 |
| `E020` | ResNet-18 (Proposed) | `lr=0.001, bs=64, wd=0.0001, epochs=10` | 42 | 0.9868 | — | 2.43 | ResNet-18 tuning trial 8/12 |
| `E021` | ResNet-18 (Proposed) | `lr=0.003, bs=32, wd=1e-05, epochs=10` | 42 | 0.9567 | — | 2.89 | ResNet-18 tuning trial 9/12 |
| `E022` | ResNet-18 (Proposed) | `lr=0.0003, bs=32, wd=0.0001, epochs=10` | 42 | 0.9886 | — | 2.87 | ResNet-18 tuning trial 10/12 |
| `E023` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=10` | 42 | **0.9893** | — | 2.39 | Selected ResNet-18 config (trial 11/12) |
| `E024` | ResNet-18 (Proposed) | `lr=0.0003, bs=32, wd=1e-05, epochs=10` | 42 | 0.9872 | — | 2.87 | ResNet-18 tuning trial 12/12 |
| `E025` | Majority-Class Baseline | `strategy=most_frequent` | 42 | 0.0239 | 0.0167 | 0.00 | Final multi-seed run (Seed 42) |
| `E026` | PCA + Random Forest | `n_estimators=400, max_depth=None, max_features='sqrt'` | 42 | 0.7024 | 0.4349 | 0.12 | Final multi-seed run (Seed 42) |
| `E027` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=15` | 42 | 0.9905 | 0.8599 | 3.65 | Final multi-seed run (Seed 42) |
| `E028` | Majority-Class Baseline | `strategy=most_frequent` | 100 | 0.0239 | 0.0167 | 0.00 | Final multi-seed run (Seed 100) |
| `E029` | PCA + Random Forest | `n_estimators=400, max_depth=None, max_features='sqrt'` | 100 | 0.6974 | 0.4362 | 0.12 | Final multi-seed run (Seed 100) |
| `E030` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=15` | 100 | 0.9893 | 0.8506 | 3.67 | Final multi-seed run (Seed 100) |
| `E031` | Majority-Class Baseline | `strategy=most_frequent` | 2026 | 0.0239 | 0.0167 | 0.00 | Final multi-seed run (Seed 2026) |
| `E032` | PCA + Random Forest | `n_estimators=400, max_depth=None, max_features='sqrt'` | 2026 | 0.6977 | 0.4397 | 0.12 | Final multi-seed run (Seed 2026) |
| `E033` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=15` | 2026 | 0.9897 | 0.8515 | 3.64 | Final multi-seed run (Seed 2026) |
| `E034` | Majority-Class Baseline | `strategy=most_frequent` | 777 | 0.0239 | 0.0167 | 0.00 | Final multi-seed run (Seed 777) |
| `E035` | PCA + Random Forest | `n_estimators=400, max_depth=None, max_features='sqrt'` | 777 | 0.6981 | 0.4402 | 0.12 | Final multi-seed run (Seed 777) |
| `E036` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=15` | 777 | 0.9911 | 0.8554 | 3.63 | Final multi-seed run (Seed 777) |
| `E037` | Majority-Class Baseline | `strategy=most_frequent` | 999 | 0.0239 | 0.0167 | 0.00 | Final multi-seed run (Seed 999) |
| `E038` | PCA + Random Forest | `n_estimators=400, max_depth=None, max_features='sqrt'` | 999 | 0.7029 | 0.4357 | 0.11 | Final multi-seed run (Seed 999) |
| `E039` | ResNet-18 (Proposed) | `lr=0.0003, bs=64, wd=1e-05, epochs=15` | 999 | 0.9910 | 0.8633 | 3.64 | Final multi-seed run (Seed 999) |
| `E040` | ResNet-18 (Sensitivity LR) | `lr=0.0001, epochs=10` | 42 | 0.9850 | — | 2.41 | Learning rate sweep |
| `E041` | ResNet-18 (Sensitivity LR) | `lr=0.0003, epochs=10` | 42 | 0.9854 | — | 2.42 | Learning rate sweep |
| `E042` | ResNet-18 (Sensitivity LR) | `lr=0.001, epochs=10` | 42 | 0.9857 | — | 2.40 | Learning rate sweep |
| `E043` | ResNet-18 (Sensitivity LR) | `lr=0.003, epochs=10` | 42 | 0.9722 | — | 2.40 | Learning rate sweep |
| `E044` | Ablation: w/o Data Augmentation | `pretrained=True, augment=False, class_weighted=True, img_size=64` | 42 | 0.9870 | 0.8586 | 3.32 | Remove flips and rotations |
| `E045` | Ablation: w/o Class-Weighted Loss | `pretrained=True, augment=True, class_weighted=False, img_size=64` | 42 | 0.9887 | 0.8533 | 3.76 | Unweighted cross-entropy |
| `E046` | Ablation: w/o ImageNet Pretraining | `pretrained=False, augment=True, class_weighted=True, img_size=64` | 42 | 0.9598 | 0.7120 | 3.76 | Random initialization |
| `E047` | Ablation: Downsampled Input ($16 \times 16$) | `pretrained=True, augment=True, class_weighted=True, img_size=16` | 42 | 0.8988 | 0.6072 | 3.61 | Match RF $16 \times 16$ spatial resolution |

---

## **Part C: Reproducible Results Package**

All code, CSV outputs, figures, and metadata are organized in our repository:
- **Pipeline Script:** `src/run_milestone5.py`
- **Summary & Seed Metrics:** `outputs/m5_main_results.csv`, `outputs/m5_raw_per_seed.csv`, `outputs/m5_uncertainty_stats.json`, `outputs/m5_tuning_records.csv`
- **Error, Sensitivity, Robustness & Ablation Data:** `outputs/m5_per_class_f1.csv`, `outputs/m5_error_analysis_samples.csv`, `outputs/m5_border_buffer_sensitivity.csv`, `outputs/m5_robustness.csv`, `outputs/m5_ablation.csv`
- **Figures:** `outputs/m5_fig1_model_comparison_ci.png`, `outputs/m5_confusion_matrices.png`, `outputs/m5_fig2_lr_sensitivity.png`
- **Run Log & Environment:** `outputs/m5_experiment_log.csv`, `outputs/m5_environment_meta.json`


