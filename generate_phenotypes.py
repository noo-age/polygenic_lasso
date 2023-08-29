import torch
from pandas_plink import read_plink

(bim, fam, bed) = read_plink('data/ALL_1000G_phase1integrated_v3_impute/genotypes_genome_hapgen.controls')
genotype_df = bed.compute().T
genotype_tensor = 2-torch.tensor(genotype_df) # read_plink encodes 0/0 homozygous reference as 2, subtraction converts to # of alternate alleles

n_individuals = genotype_tensor.shape[0]
n_SNPs = genotype_tensor.shape[1]

print(n_individuals,n_SNPs)

effect_sizes = torch.tensor(n_SNPs)
