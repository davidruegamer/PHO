############## Script for benchmark data preprocessing ################
library(readr)
library(readxl)
library(reticulate)
sklearn <- import("sklearn.datasets")

##### Airfoil #####

airfoil <- read.table("original_data/airfoil_self_noise.dat")
write.table(airfoil, file = "benchmark_data/airfoil.data", col.names = FALSE, row.names = FALSE)

##### Concrete #####

concrete <- read_excel("original_data/Concrete_Data.xls")
write.table(as.data.frame(concrete), file = "benchmark_data/concrete.data", col.names = FALSE, row.names = FALSE)

##### Diabetes #####

diabetes <- sklearn$load_diabetes(return_X_y=TRUE)
write.table(as.data.frame(cbind(diabetes[[1]],diabetes[[2]])), 
            file = "benchmark_data/diabetes.data", col.names = FALSE, row.names = FALSE)

##### Energy #####

energy <- read_excel("original_data/energy.xlsx")
write.table(energy[,-ncol(energy)], file = "benchmark_data/energy.data", col.names = FALSE, row.names = FALSE)

##### Forest Fire #####

forest_fire <- read_csv("original_data/forestfires.csv")
# transform outcome as described in 
# the repository (https://archive.ics.uci.edu/ml/datasets/forest+fires)
forest_fire$area <- scale(log(forest_fire$area + 1), scale = F)
# date conversion
forest_fire$month <- as.numeric(factor(forest_fire$month, levels = c("jan", "feb", "mar",
                                                                        "apr", "may", "jun", 
                                                                        "jul", "aug", "sep", 
                                                                        "oct", "nov", "dec")))
forest_fire$day <- as.numeric(factor(forest_fire$day, levels = c("mon", "tue", "wed", "thu", "fri", "sat", "sun")))
write.table(forest_fire, file = "benchmark_data/forest_fire.data", col.names = FALSE, row.names = FALSE)

##### Naval #####

naval <- read.table("original_data/naval.txt")
# drop features with no variance
naval$V9 <- NULL
naval$V12 <- NULL
# use only response
write.table(naval[,-ncol(naval)], file = "benchmark_data/naval_compressor.data", col.names = FALSE, row.names = FALSE)
write.table(naval[,-(ncol(naval)-1)], file = "benchmark_data/naval_turbine.data", col.names = FALSE, row.names = FALSE)

##### Wine #####

wine <- sklearn$load_wine(return_X_y=TRUE)
write.table(as.data.frame(cbind(wine[[1]],wine[[2]])),
            file = "benchmark_data/wine.data", col.names = FALSE, row.names = FALSE)

##### Yacht #####

yacht <- read.table("original_data/yacht.data", header = FALSE)
write.table(yacht, file = "benchmark_data/yacht.data", col.names = FALSE, row.names = FALSE)
