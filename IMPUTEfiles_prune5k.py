import pandas as pd

for i in range(1,23):
    # Read the files
    genetic_map = pd.read_csv(f'data/ALL_1000G_phase1integrated_v3_impute/genetic_map_chr{i}_combined_b37.txt', sep=' ', skiprows=1, names=['position', 'COMBINED_rate(cM/Mb)', 'Genetic_Map(cM)'])
    legend = pd.read_csv(f'data/ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3_chr{i}_impute.legend', sep=' ', skiprows=1,names=['id', 'position', 'a0', 'a1', 'afr.aaf', 'amr.aaf', 'asn.aaf', 'eur.aaf', 'afr.maf', 'amr.maf', 'asn.maf', 'eur.maf'])
    hap = pd.read_csv(f'data/ALL_1000G_phase1integrated_v3_impute/chr{i}.ceu_subset.hap', sep=' ', header=None)

    # MAF < 0.01 filter 
    legend = legend[legend['eur.maf'] >= 0.01]

    # Filter the first 2000 SNPs that exist in both the genetic map file and the .legend file
    common_positions = pd.merge(genetic_map, legend, on='position', how='inner')['position'][:5000]

    # Get the rows from the .legend file and the genetic map
    filtered_legend = legend[legend['position'].isin(common_positions)]
    filtered_genetic_map = genetic_map[genetic_map['position'].isin(common_positions)]

    # Get the corresponding rows from the .hap file
    filtered_hap = hap.loc[filtered_legend.index]

    # Save the files
    filtered_genetic_map.to_csv(f'data/ALL_1000G_phase1integrated_v3_impute/genetic_map_chr{i}_combined_b37_pruned.txt', sep=' ', index=False)
    filtered_legend.to_csv(f'data/ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3_chr{i}_impute_pruned.legend', sep=' ', index=False)
    filtered_hap.to_csv(f'data/ALL_1000G_phase1integrated_v3_impute/chr{i}.ceu_subset_pruned.hap', sep=' ', header=False, index=False)