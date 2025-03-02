# Pairwise-Importance-Estimate-Extension-with-Time-Series-adaptation

## Datasets
- Univariate Simulated Dataset 0
- Univariate Simulated Dataset 1
- Multivariate Simulated Dataset 0
- Multivariate Simulated Dataset 1
- [Occupancy](https://archive.ics.uci.edu/dataset/357/occupancy+detection)
- [HHAR](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)

## Methods
- Ablation study: Retraining with Singleton feature subsets & Retraining with Leave-One-Out feature subsets.
- [Deep Feature Selection](https://www.researchgate.net/publication/271329170_Deep_feature_selection_Theory_and_application_to_identify_enhancers_and_promoters) (DF) with Element wise Multiplication adaptation and Hardmard Product adapation.
- [Neural based pairwise structure](https://www.sciencedirect.com/science/article/pii/S0950705120304238) (NeuralFS) with Element wise Multiplication adaptation and Hardmard Product adaptation.
- [PIEE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11464895/)'s Weight based analysis (Weight-Naive) with Element wise Multiplication adaptation and Hardmard Product adaptation.
- PIEE's Gradient based analysis (Grad-AUC, Grad-ROC, Grad-STD) with Element wise Multiplication adaptation and Hardmard Product adatptation.
- [DeepLIFT](https://arxiv.org/abs/1704.02685)

*Note: multiseries signifies the use of Element wise adaptation and multivar signifies the use of Hardmard Product adaptation.*

Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
