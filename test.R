data <- read.table("mydata.ped",stringsAsFactors = FALSE)

genotypes <- as.matrix(data[,7:ncol(data)])

genotypes[genotypes == "d"] <- 0
genotypes[genotypes == "D"] <- 1

genotypes <- apply(genotypes,2,as.numeric)

effect_sizes <- runif(ncol(genotypes),-1,1)

genotype_scores <- genotypes %*% effect_sizes

phenotypes <- genotype_scores + rnorm(nrow(genotypes))

data <- cbind(genotypes,genotype_scores,phenotypes)

write.table(data, "data.txt",sep="\t",row.names=FALSE,col.names=FALSE)