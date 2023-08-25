# polygenic_lasso

This repository aims to model how noisy phenotype measurements affect polygenic predictor models. 

Empirical estimations of conditional standard error of measurement will be used to determine the noise function applied to a simulated genotype|phenotype dataset with a given genetic heritability.

The final simulated dataset will aim to have ~10k causal SNPs with additive linear effects hidden among ~5 million total SNPs. The minor allele frequencies (MAFs) of the SNPs will follow approximately the human distribution.

Various LASSO regression models will be trained and validated using k-fold cross validation under different noise functions applied to the same dataset, which aims to mimic the noise applied by different fluid intelligence tests used in practice. For all phenotypes we run for 5 different ratio weights: L1ratio∈{.1,.3,.5,.7,.9}. 

Measurements of predictor accuracy to both expressed phenotype and "true" phenotype will be made. 
