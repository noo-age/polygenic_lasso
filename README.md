# polygenic_lasso

This repository aims to model how noise from tests of fluid intelligence affects polygenic predictor models. 

Empirical measurements of the reliability coefficient and standard measurement of error (SEM) are used to determine the noise function applied to a simulated genotype|phenotype dataset.

The final simulated dataset will aim to have ~10k causal SNPs with additive linear effects hidden among ~5 million total SNPs. The minor allele frequencies (MAFs) of the SNPs will follow approximately the human distribution.

Various LASSO regression models will be trained and validated using k-fold cross validation under different noise functions applied to the same dataset, which aims to mimic the noise applied by different fluid intelligence tests used in practice.

Measurements of predictor accuracy to both expressed phenotype and "true" phenotype will be made. 
