import torch
import pandas as pd
import numpy as np
from pandas_plink import read_plink
import math
import visualize_data as vd

'''
Now normal
'''

datadir = 'data/ALL_1000G_phase1integrated_v3_impute/genotypes_genome_hapgen.controls'
outdir = 'Models/G_E_M_normal'

maf_causal_upper = 0.2
maf_causal_lower = 0.05

genetic_scalar = .7
environmental_noise_scalar = .3
CSEM_scalar = 0.1
normal_scalar = 0.17

torch.manual_seed(0)

def CSEM(phenotype, mean, std): # one-dimensional tensor
    return (CSEM_scalar * ((phenotype - mean) / std) ** 2) # standard deviations of noise at those phenotype levels

# read genotype data
(bim, fam, bed) = read_plink(datadir)
genotype_df = bed.compute().T
genotype_tensor = 2-torch.tensor(genotype_df) # read_plink encodes 0/0 homozygous reference as 2, subtraction converts to # of alternate alleles

# choose 10k causal SNPs with maf within specified bounds
maf_values = torch.sum(genotype_tensor,dim=0) / (2*genotype_tensor.shape[0])
maf_filter = ((maf_values >= maf_causal_lower) & (maf_values <= maf_causal_upper)).float()
indices_one = torch.nonzero(maf_filter == 1).view(-1)
random_indices = torch.randperm(indices_one.nelement())[:10000]# Randomly choose 10k values
maf_filter = torch.zeros_like(maf_filter) # Create a new tensor with all elements set to zero
maf_filter[indices_one[random_indices]] = 1 # Set the chosen indices to one

# data characteristics
n_individuals = genotype_tensor.shape[0]
n_SNPs = genotype_tensor.shape[1]

# simulates genetic_component tensor of size n_individuals with SD = 1, 
SNP_effect_sizes = torch.randn(n_SNPs) * maf_filter
genetic_component = torch.mv(genotype_tensor, SNP_effect_sizes) 
genetic_sd = torch.std(genetic_component)
genetic_mean = torch.mean(genetic_component)
genetic_component = genetic_scalar * genetic_component / genetic_sd

# random environmental noise with SD = 1
environmental_noise = torch.randn(n_individuals)
environmental_noise_mean = torch.mean(environmental_noise) 
environmental_noise_sd = torch.std(environmental_noise) 
environmental_noise = environmental_noise_scalar * (environmental_noise - environmental_noise_mean) / environmental_noise_sd

# true phenotype
true_phenotypes = genetic_component + environmental_noise
true_phenotypes_mean = torch.mean(true_phenotypes)
true_phenotypes_sd = torch.std(true_phenotypes)

# Define measurement error for each individual based on CSEM
# Assuming CSEM is a function that takes true phenotypes as input and returns a tensor of standard errors
#measurement_noise = CSEM(true_phenotypes, true_phenotypes_mean, true_phenotypes_sd) * torch.randn(n_individuals)
measurement_noise = torch.randn(n_individuals) * normal_scalar

# Used to see variance of noise explained by environment vs measurement noise
total_noise = environmental_noise + measurement_noise

# Add measurement error to true phenotypes to get observed phenotypes
observed_phenotypes = true_phenotypes + measurement_noise
normed_observed_phenotypes = (observed_phenotypes - torch.mean(observed_phenotypes)) / torch.std(observed_phenotypes)

# Combine everything into a dataframe
df = pd.DataFrame({
    'genetic_component': genetic_component.numpy(),
    'environmental_noise': environmental_noise.numpy(),
    'true_phenotype': true_phenotypes.numpy(),
    'measurement_noise': measurement_noise.numpy(),
    'total_noise': total_noise.numpy(),
    'observed_phenotype': observed_phenotypes.numpy(),
    'normed_observed_phenotypes': normed_observed_phenotypes.numpy()
})

df_maf = pd.DataFrame({
    'maf_values': maf_values.numpy(),
    'SNP_effect_sizes': SNP_effect_sizes.numpy()
})

# Export to CSV
df.to_csv(outdir + '/phenotypes.csv', index=False)

df_maf.to_csv(outdir + '/SNPs.csv', index=False)

print('heritabiltiy:',vd.r_squared_from_file(outdir + '/phenotypes.csv', 'genetic_component','observed_phenotype')) #heritability
print('env prop of total noise:',vd.r_squared_from_file(outdir + '/phenotypes.csv', 'environmental_noise', 'total_noise')) 
print('measurement noise prop of total noise:',vd.r_squared_from_file(outdir + '/phenotypes.csv', 'measurement_noise', 'total_noise'))