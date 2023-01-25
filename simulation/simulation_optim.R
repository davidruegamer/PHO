library(parallel)
library(tidyverse)

nr_reps_sim <- 4
nr_cores <- 4
max_epochs <- 5000

## checking distribution of random outputs



## -----------------------------------------------------------------------------------------------------
form_fun <- function(p, p_nn){ 
  

  str_feat <- paste(paste0("x", 1:p), collapse = " + ")
  str_feat_nn <- paste(paste0("x", 1:p), collapse = ", ")
  
  if(p_nn > 0){
    ustr_feat <- paste(paste0("z", 1:p_nn), collapse = ", ")
    inp_deep <- paste0(str_feat, ", ", ustr_feat)
  }else{
    inp_deep <- str_feat_nn
  }
  list(loc = as.formula(paste0("~ -1 + ",  str_feat, " + deep_model(", inp_deep, ")")), 
       scale = ~ 1)
  
}

true_coef = seq(-2.5, 2.5, by = 0.25)[-11]

additive_predictor_fun <- function(relevant_data, p, coefs = true_coef, nonlin = TRUE)
{
  
  nonlin_part <- 0
  if(nonlin) nonlin_part <- scale(rowSums(relevant_data[,(p+1):ncol(relevant_data)]), scale=F)
  as.matrix(relevant_data[,1:p])%*%coefs[1:p] + nonlin_part
  
}

true_mod_fun <- function(p, nonlin){
  
  form <- as.formula(paste0("y ~ 0 + ", paste(paste0("x", 1:p), collapse = " + ")))
  return(function(data) lm(form, data = data))
  
}

deep_mod_fun <- function(size, activ = "relu", last_hidden_units = 50){
  
  if(size == 1){
    
    deep_model <- function(x)
    {
      x %>% 
        layer_dense(units = last_hidden_units, activation = activ, use_bias = FALSE) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "last_layer")
    }
    
  }else if(size == 3){
    
    deep_model <- function(x)
    {
      x %>% 
        layer_dense(units = 100, activation = activ, use_bias = FALSE) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = last_hidden_units, activation = activ) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "last_layer")
    }
    
  }else if(size == 5){
    
    deep_model <- function(x)
    {
      x %>% 
        layer_dense(units = 200, activation = activ, use_bias = FALSE) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 100, activation = activ) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = last_hidden_units, activation = activ) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "last_layer")
    }
    
  }
  
  return(deep_model)
  
}

## -----------------------------------------------------------------------------------------------------

settings <- expand.grid(
  sizes = c(1, #3, 
            5),
  p = c(20),
  p_nn_in = c(#1, 
    10, 20),
  activ = c("relu"#, "tanh"
            ),
  last_hidden_units = c(#20, 
    50),
  bs = c(50#, 1000
         ),
  nonlin = c(FALSE, TRUE),
  n = c(500, 1000)
)

set.seed(42)
data <- as.data.frame(
  matrix(rnorm(max(settings$n) * (max(settings$p) + max(settings$p_nn_in))), 
         nrow = max(settings$n), ncol = max(settings$p) + max(settings$p_nn_in))
)
colnames(data) <- c(paste0("x", 1:max(settings$p)),
                    paste0("z", 1:max(settings$p_nn_in)))

newdata <- as.data.frame(
  matrix(rnorm(max(settings$n) * (max(settings$p) + max(settings$p_nn_in))), 
         nrow = max(settings$n), ncol = max(settings$p) + max(settings$p_nn_in))
)
colnames(newdata) <- c(paste0("x", 1:max(settings$p)),
                       paste0("z", 1:max(settings$p_nn_in)))

res_list <- list()

for(i in 1:nrow(settings)){
  
  n <- settings[i,]$n
  p <- settings[i,]$p
  p_nn_in <- settings[i,]$p_nn_in
  
  eta <- additive_predictor_fun(data, p, nonlin = settings[i,]$nonlin)[1:n,]
  
  args <- list(
    data = data[1:n,]
  )
    
  size <- settings[i,]$sizes
  activ <- as.character(settings[i,]$activ)
  last_hidden_units <- settings[i,]$last_hidden_units
  bs <- settings[i,]$bs
  
  Xmat <- as.matrix(data[1:n,1:p])
  
  res <- mclapply(1:nr_reps_sim, function(j){
    
    devtools::load_all("~/NSL/deepregression")
    
    set.seed(j)
    
    ### data generation
    
    outcome <- eta + rnorm(n)
    outcome <- scale(outcome, scale = FALSE)
    args$y <- outcome
    
    ### formula
    
    forms <- form_fun(p, p_nn_in)
    args$list_of_formulas <- forms
    
    ### deep model definition
    
    deep_model <- deep_mod_fun(size, activ, last_hidden_units)
    
    args$list_of_deep_models <- list(deep_model = deep_model)
    
    ### define model and fit
    
    w_oz <- orthog_control(orthogonalize = TRUE)
    wo_oz <- orthog_control(orthogonalize = FALSE)
    
    mod_w_oz <- do.call("deepregression", c(args, list(orthog_options = w_oz)))
    mod_wo_oz <- do.call("deepregression", c(args, list(orthog_options = wo_oz)))
    
    hist_ooz <- mod_w_oz %>% fit(epochs = max_epochs, early_stopping = TRUE, batch_size = bs, verbose = FALSE)
    hist_woz <- mod_wo_oz %>% fit(epochs = max_epochs, early_stopping = TRUE, batch_size = bs, verbose = FALSE)
    
    # gg_ooz <- plot(hist_ooz)
    # gg_woz <- plot(hist_woz)
    # gridExtra::grid.arrange(gg_ooz + xlim(0,1000) + ylim(0,10), gg_woz + xlim(0,1000) + ylim(0,10), ncol=2)
    
    ### apply PHO
    deep_part <- get_layer(mod_wo_oz$model, "last_layer")
    intermediate_mod <- keras_model(mod_wo_oz$model$input, deep_part$output)
    
    newdata_processed <- deepregression:::prepare_newdata(
      mod_wo_oz$init_params$parsed_formulas_contents, data[1:n,], 
      gamdata = mod_wo_oz$init_params$gamdata$data_trafos
    )
    
    zeta_wo_oz <- as.data.frame(intermediate_mod$predict(newdata_processed))
    
    # second model
    deep_part <- get_layer(mod_w_oz$model, "last_layer")
    intermediate_mod <- keras_model(mod_w_oz$model$input, deep_part$output)
    
    newdata_processed <- deepregression:::prepare_newdata(
      mod_w_oz$init_params$parsed_formulas_contents, data[1:n,], 
      gamdata = mod_w_oz$init_params$gamdata$data_trafos
    )
    
    zeta_w_oz <- as.data.frame(intermediate_mod$predict(newdata_processed))
    
    nonlin_part <- scale(rowSums(data[1:n,(p+1):ncol(data)]), scale=F)
    # plot(nonlin_part[,1], zeta_wo_oz[,1])
    # points(nonlin_part[,1], zeta_w_oz[,1], col="red")
    # abline(0,1, col="blue")
    
    pr_w_oz = predict(mod_w_oz, newdata[1:n,])[,1]
    pr_wo_oz = predict(mod_wo_oz, newdata[1:n,])[,1]
    
    fitted_w_oz = predict(mod_w_oz)[,1]
    fitted_wo_oz = predict(mod_wo_oz)[,1]
    
    wo_oz = zeta_wo_oz$V1
    w_oz  = zeta_w_oz$V1
    
    true_nonlin = nonlin_part
    true_lin = eta
    data_outcome = outcome
    newdata_outcome = (additive_predictor_fun(newdata, p, nonlin = settings[i,]$nonlin))[1:n,1]
    
    cor_lm <- lm(wo_oz ~ -1 + Xmat)
    
    ### return
    
    return(data.frame(mse_pr_wo = Metrics::mse(pr_wo_oz, newdata_outcome),
                      mse_pr_w = Metrics::mse(pr_w_oz, newdata_outcome),
                      mse_str_wo = Metrics::mse(fitted_wo_oz-wo_oz, eta),
                      mse__str_w = Metrics::mse(fitted_w_oz-w_oz, eta),
                      mse_str_pho = Metrics::mse(fitted_wo_oz-resid(cor_lm), eta),
                      mse_ustr_wo = Metrics::mse(wo_oz, true_nonlin[,1]),
                      mse_ustr_w = Metrics::mse(w_oz, true_nonlin[,1]),
                      mse_ustr_pho = Metrics::mse(resid(cor_lm), true_nonlin[,1]),
                      mse_beta_wo = Metrics::mse(unlist(coef(mod_wo_oz)), c(true_coef)),
                      mse_beta_w = Metrics::mse(unlist(coef(mod_w_oz)), c(true_coef)),
                      mse_beta_pho = Metrics::mse(unlist(coef(mod_wo_oz)) + coef(cor_lm), c(true_coef)),
                      size = size,
                      p_nn = p_nn_in,
                      activ = activ,
                      p = p,
                      bs = bs,
                      last_hidden_units = last_hidden_units,
                      n = n,
                      nonlin = settings[i,]$nonlin
                      )
    )
    
  }, mc.cores = nr_cores)
  
  res_list[[i]] <- res
  
}

resdf <- do.call("rbind", lapply(res_list, function(x) cbind(run=rep(1:nr_reps_sim, each=1), do.call("rbind", x))))

# resdf %>% group_by(size, p_nn, activ, last_hidden_units, run, bs) %>% 
#   summarise(mse_wo = Metrics::mse(pr_wo_oz, newdata_outcome),
#             mse_w = Metrics::mse(pr_w_oz, newdata_outcome)) %>% 
#   ungroup() %>% group_by(size, p_nn, activ, last_hidden_units, bs) %>% 
#   summarise(mse_wo = mean(mse_wo),
#             mse_w = mean(mse_w))
# 
# resdf %>% group_by(size, p_nn, activ, last_hidden_units, run, bs) %>% 
#   summarise(mse_wo = Metrics::mse(fitted_wo_oz-wo_oz, data_outcome-true_nonlin),
#             mse_w = Metrics::mse(fitted_w_oz-w_oz, data_outcome-true_nonlin),
#             mse_pho = Metrics::mse(fitted_wo_oz-resid(lm(wo_oz ~ -1 + Xmat)), data_outcome-true_nonlin)) %>% 
#   ungroup() %>% group_by(size, p_nn, activ, last_hidden_units, bs) %>% 
#   summarise(mse_wo = mean(mse_wo),
#             mse_w = mean(mse_w),
#             mse_pho = mean(mse_pho))
# 
# resdf %>% group_by(size, p_nn, activ, last_hidden_units, run, bs) %>% 
#   summarise(mse_wo = Metrics::mse(wo_oz, true_nonlin),
#             mse_w = Metrics::mse(w_oz, true_nonlin),
#             mse_pho = Metrics::mse(resid(lm(wo_oz ~ -1 + Xmat)), true_nonlin)) %>% 
#   ungroup() %>% group_by(size, p_nn, activ, last_hidden_units, bs) %>% 
#   summarise(mse_wo = mean(mse_wo),
#             mse_w = mean(mse_w),
#             mse_pho = mean(mse_pho))

# resdf <- resdf %>% group_by(run, size, p_nn, activ, last_hidden_units) %>% 
#   mutate(
#     PHO = resid(lm(wo_oz ~ -1 + Xmat))
#   ) %>% 
#   ungroup()

saveRDS(resdf, file="optimization_res.RDS")

if(FALSE){
  
  library(ggplot2)

  resdf %>% group_by(size, p_nn, activ, last_hidden_units, run, bs, n, nonlin) %>% 
    pivot_longer(mse_pr_wo:mse_beta_pho) %>%
    mutate(name = gsub("\\_\\_", "_", name)) %>% 
    mutate(what = gsub("mse\\_(.*)\\_(.*)", "\\1", name),
           method = gsub("mse\\_(.*)\\_(.*)", "\\2", name)) %>% 
    ggplot(aes(x=method, y=log(value), fill=method)) + 
    geom_boxplot() + facet_grid(what ~ size*p_nn*activ*p*bs*last_hidden_units*n*nonlin, scales="free_y")
    
  # colnames(resdf)[c(2:3,5)] <- c("unconstrained", "ONO","q")
  # reslong <- resdf %>% pivot_longer(c(unconstrained:ONO, PHO)) %>% 
  #   group_by(size, q, activ, p, last_hidden_units, name) %>% 
  #   summarise(value = sd(value)) %>% ungroup()
  # ggplot(reslong, 
  #        aes(x = size, y=value, colour = name)) + 
  #   geom_line() +
  #   theme_bw() + ylab("Variance of unstructured effect") + xlab("Network size") + 
  #   theme(legend.title = element_blank(),
  #         text = element_text(size = 14)) + 
  #   scale_colour_manual(values = c("#E69F00", "#999999", "#56B4E9")) + 
  #   facet_grid(activ ~ q * last_hidden_units, labeller = labeller(q = label_both)) +
  #   theme(legend.position="bottom") + scale_x_continuous(breaks = c(1,3,5))
  # 
  # ggsave(width = 6, height = 4, filename = "var_latent.pdf")
  
}

