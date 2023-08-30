import torch
import pandas as pd
from pandas_plink import read_plink

def CSEM(phenotype): # one-dimensional tensor
    return abs(0.5 * phenotype)

def r_correlation(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_centered = tensor1 - tensor1_mean
    tensor2_centered = tensor2 - tensor2_mean
    correlation = torch.sum(tensor1_centered * tensor2_centered) / (torch.sqrt(torch.sum(tensor1_centered ** 2)) * torch.sqrt(torch.sum(tensor2_centered ** 2)))
    return correlation.item()

(bim, fam, bed) = read_plink('data/ALL_1000G_phase1integrated_v3_impute/genotypes_genome_hapgen.controls')
genotype_df = bed.compute().T
genotype_tensor = 2-torch.tensor(genotype_df) # read_plink encodes 0/0 homozygous reference as 2, subtraction converts to # of alternate alleles

n_individuals = genotype_tensor.shape[0]
n_SNPs = genotype_tensor.shape[1]

SNP_effect_sizes = torch.randn(n_SNPs)

genetic_component = torch.mv(genotype_tensor, SNP_effect_sizes)
environmental_noise = torch.randn(n_individuals)
true_phenotypes = genetic_component + environmental_noise

# Define measurement error for each individual based on CSEM
# Assuming CSEM is a function that takes true phenotypes as input and returns a tensor of standard errors
measurement_errors = CSEM(true_phenotypes)

# Add measurement error to true phenotypes to get observed phenotypes
observed_phenotypes = true_phenotypes + torch.randn(n_individuals)*measurement_errors

# Combine everything into a dataframe
df = pd.DataFrame({
    'genetic_component': genetic_component.numpy(),
    'environmental_noise': environmental_noise.numpy(),
    'true_phenotype': true_phenotypes.numpy(),
    'observed_phenotype': observed_phenotypes.numpy(),
})

# Export to CSV
df.to_csv('simulated_phenotypes.csv', index=False)