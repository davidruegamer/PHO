rm(list=ls())
library(Metrics)
library(parallel)

### model functions ###
orthog_tf <- function (Y, X, stpgrad) 
{
  if(stpgrad) Q = tf$stop_gradient(tf$linalg$qr(X, full_matrices = FALSE, name = "QR")$q) else
    Q = tf$linalg$qr(X, full_matrices = FALSE, name = "QR")$q
  X_XtXinv_Xt <- tf$linalg$matmul(Q, tf$linalg$matrix_transpose(Q))
  Ycorr <- tf$subtract(Y, tf$linalg$matmul(X_XtXinv_Xt, Y))
  return(Ycorr)
}

# orthog_tf <- function (Y, X) 
# {
#   X_XtXinv_Xt <- tf$linalg$matmul(tf$linalg$matmul(X, tf$linalg$inv(tf$linalg$matmul(X, X, transpose_a=TRUE))), 
#                                   X, transpose_b=TRUE)
#   return(tf$subtract(Y, tf$linalg$matmul(X_XtXinv_Xt, Y)))
# }

model <- function(inpsize, inpsize_oz, arch, stpgrad, actfun="linear"){
  
  inp <- layer_input(as.integer(inpsize))
  im_outp <- arch(inp)
  if(inpsize_oz>0){
    inp_oz <- layer_input(as.integer(inpsize_oz))
    outp <- layer_dense(orthog_tf(im_outp, inp_oz, stpgrad), 
                        units = 1, use_bias = FALSE, activation = actfun)
    mod <- keras_model(list(inp, inp_oz), outp)
  }else{
    outp <- layer_dense(im_outp, units = 1, use_bias = FALSE,
                        activation = actfun)
    mod <- keras_model(inp, outp)
  }
  
  compile(mod, 
          optimizer = optimizer_sgd(),
          loss = "mse")
  
}

archfun <- function(size, nrhidden){
  
  function(x){
    if(size >= 1)
      for(i in 1:size)
        x <- layer_dense(x, units = nrhidden, activation = "relu", 
                         use_bias = FALSE)
    return(x)
  }
  
}

### data ###
datagen <- function(n, inpsize, inpsize_oz, sdXoz = 1,
                    nonlin = function(x) sin(x[,1]) + x[,2]^2, 
                    sdnoise = 1){
  
  X <- matrix(rnorm(n * inpsize), nrow = n)
  X_oz <- NULL
  if(inpsize_oz>0)
    X_oz <- matrix(rnorm(n * inpsize_oz, sd = sdXoz), nrow = n)
  outcome <- scale(nonlin(X) + rnorm(n, sd = sdnoise), scale = F)
  return(list(y = outcome,
                X = X, X_oz = X_oz))
  
}

### simulation ###
sim_fun <- function(setting, seed){
  
  set.seed(seed)
  
  data <- datagen(setting$n,
                  setting$inpsize,
                  setting$inpsize_oz,
                  setting$sdXoz,
                  sdnoise = setting$sdnoise
                  )
  
  arch <- archfun(setting$archsize,
                  setting$archhidden)
  
  mod <- model(setting$inpsize, 
               setting$inpsize_oz,
               arch = arch,
               stpgrad = setting$stpgrad
               )
  
  mod_inp <- if(is.null(data$X_oz))
    data$X else list(data$X, data$X_oz)
  
  mod %>% fit(
    x = mod_inp,
    y = data$y,
    batch_size = setting$batchs,
    epochs = 1000L,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 50L)
    ),
    verbose = FALSE
  )
  
  set.seed(seed + 1000)
  
  test <- datagen(setting$n,
                  setting$inpsize,
                  setting$inpsize_oz
                  )
  
  mod_inp <- if(is.null(data$X_oz))
    test$X else list(test$X, test$X_oz)

  # mod_wo_oz_post <- mod$layers[[length(mod$layers)]](
  #   get_layer(mod, "dense")$output
  # )
  #   
  # mod_nooz <- keras_model(mod$input, mod_wo_oz_post)
  # 
  # pr_nooz <- mod_nooz %>% predict(mod_inp, batch_size = 32L)
  
  pr <- mod %>% predict(
    mod_inp, batch_size = setting$pr_bs
  )
  
  return(mse(test$y, pr))
  
}

### settings ###
settings <- expand.grid(
  n = c(
    100, 
    1000, 
    10000#, 
    #1e5,
    #1e6 -> 2.089537 w OZ
  ),
  sdnoise = c(#1e-5, 1e-1, 
    1),
  inpsize = c(#2, 
    20),
  inpsize_oz = c(0, 
    10),
  batchs = c(30), #, 100, 1000),
  archsize = 1, #c(1,2,3),
  archhidden = 10, # c(3,5,10,20),
  stpgrad = c(0),#,1),
  sdXoz = c(#1e-9, 1e-6, 
    1),
  pr_bs = c(32, 1000, 10000)
)

# settings <- settings %>% 
#  filter(inpsize_oz != 0 | (inpsize_oz == 0 & stpgrad == 1))

### run ###
reps <- 4
nr_cores <- reps
res_list <- list()

for(i in 1:nrow(settings)){
  
  cat("Setting ", i, "\n")
  
  setting <- settings[i,]
  res <- mclapply(1:reps, function(r){
    library(tensorflow)
    library(keras)
    sim_fun(setting, r)
  }, mc.cores = nr_cores)
  res_list[[i]] <- res
  
}

resdf <- cbind(settings[rep(1:nrow(settings), each=reps),], rmse=unlist(res_list))

### analysis ###
library(tidyverse)
library(ggplot2)

ozdef <- function(inpsize_oz, stpgrad){
  
  case_when(
    inpsize_oz==0 & stpgrad==1 ~ "without orthog.",
    inpsize_oz==0 & stpgrad==0 ~ "without orthog.",
    inpsize_oz!=0 & stpgrad==0 ~ "with orthog.",
    inpsize_oz!=0 & stpgrad==1 ~ "with orthog. (stop gradients)"
  )
  
}

resdf %>% #group_by(across(c(-rmse))) %>% 
  # summarize(rmse = mean(rmse)) %>% 
  # arrange(-rmse) %>% 
  # print(n=32)
  mutate(
    orthog_applied = ozdef(inpsize_oz, stpgrad)
  ) %>% 
  rename(`input size` = inpsize) %>% 
  ggplot(aes(x=factor(n), y=rmse)) + 
  geom_boxplot(aes(colour = orthog_applied)) + #aes(fill=factor(batchs))) + 
  facet_grid(#`input size` 
      ~ pr_bs, #archhidden*archsize, 
      labeller = labeller(`input size` = label_both)) + 
  theme_bw() + 
  xlab("#Observations") + ylab("RMSE")

# ggsave(file="orthog_bias.pdf", width = 4, height = 5)

ggsave(file="orthog_bias_bs_adjusted.pdf", width = 4, height = 5)
