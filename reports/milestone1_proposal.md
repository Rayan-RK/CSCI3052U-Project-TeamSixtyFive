# Out-of-Distribution Generalization in Satellite Land Cover Classification: A Geographic Domain-Shift Approach
**Research Proposal**  
**CSCI 3052U — Machine Learning I  |  Team SixtyFive.AI**

## 1. Title
Out-of-Distribution Generalization in Satellite Land Cover Classification: A Geographic Domain-Shift Approach

## 2. Abstract
Standard computer vision models for satellite land cover classification typically evaluate performance using identically distributed (IID) random train/test splits. This approach creates an illusion of high accuracy while concealing severe structural fragility: models memorize local geographic features, soil colors, and building architectures, failing silently when deployed across geographic boundaries due to out-of-distribution (OOD) spatial shifts. This proposal outlines a supervised multi-class image classification pipeline using the EuroSAT multi-spectral Sentinel-2 dataset across ten land cover classes. To rigorously measure generalization, we will implement a strict spatial partitioning strategy, training models on specific geographic regions while evaluating them exclusively on spatially disjoint, held-out regions. We will implement and compare a stratified random baseline, Principal Component Analysis (PCA) combined with a Random Forest classifier, and a Convolutional Neural Network (ResNet-18 architecture). Success will be defined by quantifying the exact performance degradation ($\Delta$ Macro-$F_1$) between standard IID cross-validation and rigorous spatial OOD evaluation. The project will contribute a reproducible spatial stress-test benchmark, providing critical insights for climate scientists, urban planners, and environmental monitoring agencies that rely on globally resilient remote-sensing tools.

## 3. Research Background
Remote sensing and satellite imagery analysis underpin critical global applications, including climate change mitigation, deforestation monitoring, agricultural yield forecasting, and urban expansion planning. As high-resolution orbital imagery becomes increasingly accessible, machine learning models are routinely deployed to automate land cover classification across vast geographic scales.

However, a persistent methodological flaw in much of the applied computer vision literature is the reliance on random IID train/test splits. Because adjacent satellite image patches share high spatial autocorrelation, a random split allows patches from the exact same regional ecosystems or urban blocks to leak into both training and test sets. When these models are subsequently deployed to new countries or continents with different agricultural practices, soil composition, or architectural styles, their performance frequently degrades catastrophically.

This gap, the uncritical reliance on IID evaluation protocols versus the reality of spatial domain shift is critical to investigate. In high-stakes environmental and commercial domains, a model that provides confident but erroneous classifications due to geographic distribution shift can lead to misallocated conservation funds, unpenalized environmental violations, or flawed municipal planning. Building robust evaluation frameworks that expose these vulnerabilities is essential for trustworthy automated remote sensing.

## 4. Methodology
* **Data:** We will use the EuroSAT dataset, an open-access collection of 27,000 labeled multi-spectral Sentinel-2 satellite image patches spanning 10 distinct land cover classes (e.g., Annual Crop, Forest, Herbaceous Vegetation, Industrial, Residential, River). The dataset is approximately 2 GB, making it fully processable on standard hardware or cloud-hosted research environments.
* **Methods:** To eliminate spatial data leakage, our pipeline will reject random splitting in favor of a Geographic Spatial Partitioning Strategy. We will segregate training data and evaluation data by distinct geographic zones (training on Western/Central European imagery subsets and testing on Southern/Eastern European subsets). For the classical baseline, we will extract flattened pixel arrays, apply Principal Component Analysis (PCA) for dimensionality reduction, and train a Random Forest classifier. For the deep learning approach, we will fine-tune a lightweight Convolutional Neural Network (ResNet-18) adapted for multi-spectral input channels.
* **Experiments and Evaluation:** We will train and compare three tiers of models: (1) a naive stratified random baseline; (2) a PCA + Random Forest pipeline as our structured feature baseline; and (3) a fine-tuned ResNet-18 CNN. Hyperparameters will be optimized using cross-validation restricted strictly to the training geographic zones.
* **Evaluation Metrics:** The primary metric is Macro-$F_1$ score, evaluated both under standard random cross-validation and across the spatial OOD test boundary. We will analyze confusion matrices to identify which specific land cover classes (e.g., confusing pastures with industrial concrete) are most vulnerable to geographic shift.
* **Theoretical Comparison:** We will ground our results by contrasting the mathematical foundations of our models: the covariance matrix eigenvalue decomposition underlying the PCA baseline versus the translational invariance and hierarchical spatial filters learned by the convolutional layers of the ResNet architecture.

## 5. Significance
This research will establish a reproducible, leakage-free spatial evaluation benchmark for satellite imagery classification, moving beyond over-optimistic IID accuracy metrics. Climate scientists, remote sensing engineers, and geographic information systems (GIS) professionals will benefit directly from an honest appraisal of how modern models behave under real-world domain shifts. By highlighting the exact failure modes of land cover classifiers across borders, this project provides a rigorous template for developing safer, more reliable geospatial AI systems.

## 6. SMART Timeline
| Milestone | Target Date | Deliverable |
| :--- | :--- | :--- |
| **Milestone 1 - Proposal & Team Contract** | Sept. 18–20 | Finalize proposal, sign team contract, initialize private GitHub repository with initial directory structure. |
| **Milestone 2 - Dataset & Problem Approval** | Oct. 2 | Complete data card, document class distributions, define spatial train/OOD-test geographic boundaries. |
| **Milestone 3 - Baseline Reproduction** | Oct. 19 | Build data pipeline, implement PCA + Random Forest baseline, establish reproducible environment, document discrepancies. |
| **Milestone 4 - Progress Presentation** | Oct. 30 | Complete theoretical justification for ResNet vs. PCA, present preliminary IID vs. OOD results, address blockers. |
| **Milestone 5 - Final Experiments & Analysis** | Nov. 20 | Finalize CNN hyperparameter tuning, execute robustness and class-level error analysis tables. |
| **Milestone 6 - Final Report & Demonstration** | Dec. 4 | Deliver final presentation, publish reproducible repository, submit final report. |

## 7. References
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
* Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226.
* Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572.
* Roscher, R., Bohn, B., Duarte, M. F., & Garcke, J. (2020). Explainable machine learning for remote sensing. *IEEE Geoscience and Remote Sensing Magazine*, 8(4), 36–49.
