# polygenic_lasso

This repository aims to model how noisy phenotype measurements affect polygenic predictor models. 

Empirical estimations of conditional standard error of measurement will be used to determine the noise function applied to a simulated genotype|phenotype dataset with a given genetic heritability.

The final simulated dataset will aim to have ~10k causal SNPs with additive linear effects hidden among ~50k total SNPs after pruning. Causal variants are chosen to have minor allele frequencies between 0.05 and 0.2.

Various LASSO regression models will be trained and validated using k-fold cross validation under different noise functions applied to the same dataset, which aims to mimic the noise applied by different fluid intelligence tests used in practice.

Measurements of predictor accuracy to both expressed phenotype and "true" phenotype will be made. 
