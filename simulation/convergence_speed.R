rm(list=ls())
library(Metrics)
library(parallel)

### model functions ###
orthog_tf <- function (Y, X) 
{
  Q = tf$linalg$qr(X, full_matrices = FALSE, name = "QR")$q
  X_XtXinv_Xt <- tf$linalg$matmul(Q, tf$linalg$matrix_transpose(Q))
  Ycorr <- tf$subtract(Y, tf$linalg$matmul(X_XtXinv_Xt, Y))
  return(Ycorr)
}

model <- function(inpsize, inpsize_oz, arch, actfun="linear"){
  
  inp <- layer_input(as.integer(inpsize))
  im_outp <- arch(inp)
  if(inpsize_oz>0){
    inp_oz <- layer_input(as.integer(inpsize_oz))
    outp <- layer_dense(orthog_tf(im_outp, inp_oz), 
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
  
  if(size==1) names <- "layer_before_OZ" else
    names <- c(letters[1:(size-1)], "layer_before_OZ")
  
  function(x){
    if(size >= 1)
      for(i in 1:size)
        x <- layer_dense(x, units = nrhidden, activation = "relu", 
                         use_bias = FALSE, name = names[i])
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
               arch = arch
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
  
  # prediction w/o OZ
  mod_inp <- if(is.null(data$X_oz))
    test$X else list(test$X, test$X_oz)

  # prediction with OZ
  pr <- mod %>% predict(
    mod_inp, batch_size = setting$pr_bs
  )
    
  mod_wo_oz_post <- mod$layers[[length(mod$layers)]](
    get_layer(mod, "layer_before_OZ")$output
  )

  mod_nooz <- keras_model(mod$input, mod_wo_oz_post)

  pr_nooz <- mod_nooz %>% predict(mod_inp, batch_size = 32L)
  
  return(c(mse(test$y, pr),mse(test$y, pr_nooz)))
  
}

### settings ###
settings <- expand.grid(
  n = 10^(2:5),
  sdnoise = c(1),
  inpsize = c(20),
  inpsize_oz = c(10),
  batchs = c(32),
  archsize = 1, 
  archhidden = 10, 
  sdXoz = 1,
  pr_bs = 10^(0:4)
)

# settings <- settings %>% 
#  filter(inpsize_oz != 0 | (inpsize_oz == 0 & stpgrad == 1))

### run ###
reps <- 4
nr_cores <- reps
withOZ_list <- list()
woOZ_list <- list()

for(i in 1:nrow(settings)){
  
  cat("Setting ", i, "\n")
  
  setting <- settings[i,]
  res <- mclapply(1:reps, function(r){
    library(tensorflow)
    library(keras)
    sim_fun(setting, r)
  }, mc.cores = nr_cores)
  withOZ_list[[i]] <- sapply(res, "[[", 1)
  woOZ_list[[i]] <- sapply(res, "[[", 2)
  
}

resdf <- cbind(settings[rep(1:nrow(settings), each=reps),], 
               rmse_wOZ=unlist(withOZ_list), rmse_woOZ=unlist(woOZ_list))

saveRDS(resdf, file="limitation_pe.RDS")

### analysis ###
library(tidyverse)
library(ggplot2)


resdf %>%
  rename(`with Projection` = rmse_wOZ,
         `without Projection` = rmse_woOZ,
         m = pr_bs) %>% 
  # mutate(`with Orthog.` = 1+1*(inpsize_oz > 0)) %>% 
  pivot_longer(`with Projection`:`without Projection`) %>% 
  rename(`input size` = inpsize) %>% 
  group_by(n, m, name) %>%
  summarize(value = mean(value)) %>%
  filter(n>100) %>% 
  ggplot(aes(x=m, y=value)) + 
  geom_hline(yintercept = 1.5, linetype = 2, alpha = 0.7) + 
  # geom_boxplot(aes(colour = name)) + 
  # geom_smooth(aes(x = n, y = value, colour = name)) + 
  geom_line(aes(x = m, y = value, colour = name), lwd=1.2) + 
  # geom_boxplot(aes(colour = `with Orthog.`)) + #aes(fill=factor(batchs))) + 
  facet_grid(~ n, #archhidden*archsize, 
    labeller = labeller(n = label_both)
    ) + 
  theme_bw() + 
  theme(legend.title = element_blank(),
        legend.position = "bottom",
        text = element_text(size = 15),
        panel.spacing = unit(1.2, "lines"),
        axis.text.x = element_text(angle = 40, hjust = 1, size = 12)) +
  xlab("Batch Size b") + ylab("RMSE") + 
  scale_x_continuous(trans = "log10",
                     labels = function(x) format(x, scientific = TRUE)) 
  

ggsave(file="orthog_bias_bs_adjusted.pdf", width = 7.5, height = 3.3)

resdf %>%
  rename(`with Proj.` = rmse_wOZ,
         `w/o Proj.` = rmse_woOZ,
         m = pr_bs) %>% 
  # mutate(`with Orthog.` = 1+1*(inpsize_oz > 0)) %>% 
  pivot_longer(`with Proj.`:`w/o Proj.`) %>% 
  rename(`input size` = inpsize) %>% 
  group_by(n, m, name) %>%
  summarize(value = mean(value)) %>%
  filter(n==1e+5) %>% 
  pivot_wider(names_from = name, values_from = value) %>% 
  summarise(pred_error = `with Proj.`-`w/o Proj.`) %>% 
  filter(m > 1) -> resdf_pe 

plot(log(resdf_pe$pred_error^2) ~ log(resdf_pe$m^2))
lm(log(pred_error^2) ~ log(m^2), data=resdf_pe)
