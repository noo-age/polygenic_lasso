# polygenic_lasso

This repository aims to model how noisy phenotype measurements affect polygenic predictor models. 

Using hapgen2_plink_simulation_workflow.sh will give PLINK format (bed) genomic data with n = 10,000 and ~50,000 SNPs for each individual, with minor linkage disequilibrium pruning. Phenotypes can be generated using generate_phenotypes.py, which simulates phenotypes with a genetic component, environmental component, and measurement noise component. The genetic component is modeled using ~10k SNPs with linear, normally distributed effect sizes out of the ~50k measured for each individual in the previously generated PLINK format dataset. The minor allele frequency of causal SNPs is set to be between 0.05 and 0.2. Output is in form of two files, phenotypes.csv and SNPs.csv. phenotypes.csv contains phenotype data (measured in z-score) as well as the individual breakdown of genetic, environmental, and measurement components. SNPs.csv lists all 50k SNPs' minor allele frequencies and simulated effect sizes. 

lasso.py contains a LASSO regression model written in PyTorch with a training script that uses k-fold cross-validation. Model predictions on the validation data are saved at the end of the training run for each model and saved to a csv file. Losses are appended to a csv files as well. 

