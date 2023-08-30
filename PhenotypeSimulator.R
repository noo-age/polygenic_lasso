# define directories
indir <- '/home/yejun/code/polygenic_lasso/data/ALL_1000G_phase1integrated_v3_impute'
datadir <- '/home/yejun/code/polygenic_lasso/data/phenotypes'
if (!dir.exists(datadir)) dir.create(datadir, recursive=TRUE)

# specify filenames and parameters 
totalGeneticVar <- 0.4
totalSNPeffect <- 0.1
h2s <- totalSNPeffect/totalGeneticVar
kinshipfile <- paste(indir, "/genotypes_genome_hapgen.controls.grm.rel", sep="")

genoFilePrefix <- paste(indir, "/genotypes_", sep="")
genoFileSuffix <- "_hapgen.controls.gen"

'''
Note, command below apparently needs this, file naming issue

#for i in {1..22}
do
    mv genotypes_chr${i}_hapgen.controls.sample genotypes_chr${i}_hapsample.controls.sample
done
'''

# simulate phenotype with three phenotype components
simulation <- runSimulation(N = 100, P = 3, cNrSNP=10000, seed=43,
                           kinshipfile = kinshipfile,
                           format = "oxgen",
                           genoFilePrefix = genoFilePrefix,
                           genoFileSuffix = genoFileSuffix,
                           chr = 1:22,
                           mBetaGenetic = 0, sdBetaGenetic = 0.2,
                           theta=1,
                           NrSNPsOnChromosome=rep(500,22),
                           genVar = totalGeneticVar, h2s = h2s,
                           phi = 0.6, delta = 0.2, rho=0.2,
                           NrFixedEffects = 2, NrConfounders = c(2, 2),
                           distConfounders = c("bin", "norm"),
                           probConfounders = 0.2,
                           genoDelimiter=" ",
                           kinshipDelimiter="\t",
                           kinshipHeader=FALSE,
                           verbose = TRUE )

savedir <- paste(system.file("extdata", package="PhenotypeSimulator"), 
                            "/resultsSimulationAndLinearModel", sep="")
saveRDS(simulation$phenoComponentsFinal, 
        paste(savedir, "/simulation_phenoComponentsFinal.rds", sep=""))
                           