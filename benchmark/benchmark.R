############################# Loading libraries #############################
library(caret)
library(parallel)
nr_cores <- 10
tryNA <- function(expr) tryCatch(expr, error = function(e) NA)
############################# Data loader #################################
data_reader <- function(
  name = c("airfoil", "concrete", "diabetes", "energy",
           "forest_fire", "naval_compressor", "yacht") 
    ){

    name <- match.arg(name)
    data <- read.table(paste0("benchmark_data/", name, ".data"))
    return(data)
  
}
########################### Benchmark function #############################
benchmark_per_dataset <- function(name, folds = 10){
  
  data <- data_reader(name)
  
  X <- data[,1:(ncol(data)-1)]
  # exclude columns with not enough unique values
  X <- X[,which(apply(X, 2, function(x) length(unique(x)))>10)]
  X <- as.data.frame(scale(X))
  y <- data[,ncol(data)]
  
  set.seed(1)
  folds <- createFolds(y, k = folds)
  
  res <- mclapply(folds, function(testind){
    
    # source models within apply to allow for parallelization
    source("models.R")
    
    # data
    trainind <- setdiff(1:nrow(X), testind)
    trainX <- X[trainind,]
    trainY <- as.numeric(y[trainind])
    testX <- X[testind,]
    testY <- as.numeric(y[testind])
    
    # models
    res_fold_i <- suppressMessages(suppressWarnings(
      data.frame(
        gam = RMSE(tryNA(gamr(trainX, trainY, testX)), testY),
        ono_a = RMSE(tryNA(ono(architecture = "a", trainX, trainY, testX)), testY),
        ono_b = RMSE(tryNA(ono(architecture = "b", trainX, trainY, testX)), testY),
        ono_c = RMSE(tryNA(ono(architecture = "c", trainX, trainY, testX)), testY), # = same as ono_d
        pho_a = RMSE(tryNA(pho(architecture = "a", trainX, trainY, testX)), testY),
        pho_b = RMSE(tryNA(pho(architecture = "b", trainX, trainY, testX)), testY),
        pho_c = RMSE(tryNA(pho(architecture = "c", trainX, trainY, testX)), testY),
        pho_d = RMSE(tryNA(pho(architecture = "d", trainX, trainY, testX)), testY),
        mlp_a = RMSE(tryNA(mlp(architecture = "a", trainX, trainY, testX)), testY),
        mlp_b = RMSE(tryNA(mlp(architecture = "b", trainX, trainY, testX)), testY),
        mlp_c = RMSE(tryNA(mlp(architecture = "c", trainX, trainY, testX)), testY),
        mlp_d = RMSE(tryNA(mlp(architecture = "d", trainX, trainY, testX)), testY)
        )
    ))
    
    
    
  }, mc.cores = nr_cores)
  
  return(res)
  
}

if(!dir.exists("results"))
  dir.create("results")

datas <- c("airfoil", "concrete", "diabetes", "energy",
           "forest_fire", "naval_compressor",  "yacht") 

for(nam in datas){
  
  res <- benchmark_per_dataset(nam)
  saveRDS(res, file=paste0("results/", nam, ".RDS"))
  
}

# produce result table
library(xtable)
library(dplyr)
rounding <- 2
lf <- list.files("results/")
tab_raw <- do.call("rbind", lapply(1:length(lf), function(i){
  
  table_for_data_i <- do.call("rbind", readRDS(paste0("results/", lf[i])))
  if(lf[i]=="naval_compressor.RDS") table_for_data_i <- table_for_data_i * 100
  means <- apply(table_for_data_i, 2, mean)
  sds <- apply(table_for_data_i, 2, sd)
  df <- as.data.frame(t(paste0(signif(means, rounding), " (", signif(sds, rounding), ")")))
  rownames(df) <- gsub("(.*)\\.RDS", "\\1", lf[i])
  colnames(df) <- names(means)
  return(df)
  
})) 

dsn <- tools::toTitleCase(rownames(tab_raw))
dsn[dsn=="Forest_fire"] <- "ForestF"
dsn[dsn=="Naval_compressor"] <- "Naval"

rownames(tab_raw) <- dsn

tab_raw %>% xtable()

tab_agg <- 
  data.frame(GAM = tab_raw$gam, 
             `MLP_l` = apply(tab_raw[,c("mlp_a","mlp_b")],1,min),
             `MLP_s` = apply(tab_raw[,c("mlp_c","mlp_d")],1,min),
             `ONO_l` = apply(tab_raw[,c("ono_a","ono_b")],1,min),
             `ONO_s` = tab_raw$ono_c,
             `PHO_l` = apply(tab_raw[,c("pho_a","pho_b")],1,min),
             `PHO_s` = apply(tab_raw[,c("pho_c","pho_d")],1,min)
             )

tab_agg %>% t() %>% xtable()

