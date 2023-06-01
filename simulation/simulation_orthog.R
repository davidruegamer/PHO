library("mgcv")
library("parallel")

## -----------------------------------------------------------------------------------------------------
settings <- expand.grid(
  n = c(100, 1000),
  p = c(1, 3, 10),
  p_nn = c(0, 30),
  nonlin = c(0, 1)
)

set.seed(42)
data <- as.data.frame(
  matrix(rnorm(max(settings$n) * (max(settings$p) + max(settings$p_nn))),
         nrow = max(settings$n), ncol = max(settings$p) + max(settings$p_nn))
)
colnames(data) <- c(paste0("x", 1:max(settings$p)),
                    paste0("z", 1:max(settings$p_nn)))

nr_reps_sim <- 20
nr_cores <- 4
max_epochs <- 5000

lotf <- list(function(x) cos(5*x),
             function(x) tanh(3*x),
             function(x) -x^3,
             function(x) cos(x*3-2)*(-x*3),
             function(x) exp(x*0.5) - 1,
             function(x) x^2,
             function(x) sin(x)*cos(x),
             function(x) sqrt(abs(x)),
             function(x) dnorm(x)-0.125,
             function(x) -x * tanh(3*x) * sin(4*x))

## -----------------------------------------------------------------------------------------------------
form_fun <- function(p, p_nn, nonlin, edfs){ 
  
  if(nonlin){
    if(is.null(edfs)){
      str_feat <- paste(paste0("s(x", 1:p, ")"), collapse = " + ")
    }else{
      str_feat <- paste(paste0("s(x", 1:p, ", df = ", round(edfs,5), ")"), collapse = " + ")
    }
  }else{
    str_feat <- paste(paste0("x", 1:p), collapse = " + ")
  }
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

additive_predictor_fun <- function(relevant_data, p, nonlin,  
                                   functions = lotf,
                                   coefs = seq(-2.5, 2.5, by = 0.5)[-6]
)
{
  
  if(!nonlin) return(as.matrix(relevant_data[,1:p])%*%coefs[1:p])
  return(matrix(rowSums(sapply(1:p, function(i) functions[[i]](relevant_data[,i])))))
  
}

true_mod_fun <- function(p, nonlin){
  
  if(nonlin){
    form <- as.formula(paste0("y ~ 0 + ", paste(paste0("s(x", 1:p, ")"), collapse = " + ")))
    return(function(data) gam(form, data = data))
  }else{
    form <- as.formula(paste0("y ~ 0 + ", paste(paste0("x", 1:p), collapse = " + ")))
    return(function(data) lm(form, data = data))
  }
  
}

## -----------------------------------------------------------------------------------------------------

res_list <- list()

for(i in 1:nrow(settings)){
  
  n <- settings$n[i]
  p <- settings$p[i]
  p_nn <- settings$p_nn[i]
  nonlin <- settings$nonlin[i]
  
  ind_cols <- c(1:p)
  if(p_nn>0) ind_cols <- c(ind_cols, max(settings$p)+1:p_nn)
  
  relevant_data <- data[1:n, ind_cols, drop=FALSE]
  
  eta <- additive_predictor_fun(relevant_data, p, nonlin)
  
  args <- list(
    data = relevant_data
  )
  
  res <- mclapply(1:nr_reps_sim, function(j){
    
    library(deepregression)
    
    set.seed(j)
    
    ### data generation
    
    outcome <- eta + rnorm(n)
    outcome <- scale(outcome, scale = FALSE)
    args$y <- outcome
    
    ### GAM model
    
    relevant_data$y <- outcome
    true_mod <- true_mod_fun(p, nonlin)(relevant_data)
    edfs <- NULL
    if(nonlin) edfs <- sapply(1:length(true_mod$smooth), function(i) 
      sum(true_mod$edf[true_mod$smooth[[i]]$first.para:true_mod$smooth[[i]]$last.para]))
    
    ### formula
    
    forms <- form_fun(p, p_nn, nonlin, edfs = edfs)
    args$list_of_formulas <- forms
    
    ### deep model definition
    
    deep_model <- function(x)
    {
      x %>% 
        layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 50, activation = "relu") %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "last_layer")
    }
    
    args$list_of_deep_models <- list(deep_model = deep_model)
    
    ### define model and fit
    
    w_oz <- orthog_control(orthogonalize = TRUE)
    wo_oz <- orthog_control(orthogonalize = FALSE)
    
    mod_w_oz <- do.call("deepregression", c(args, list(orthog_options = w_oz)))
    mod_wo_oz <- do.call("deepregression", c(args, list(orthog_options = wo_oz)))
    
    hist_ooz <- mod_w_oz %>% fit(epochs = max_epochs, early_stopping = TRUE, batch_size = 50, verbose = FALSE)
    hist_woz <- mod_wo_oz %>% fit(epochs = max_epochs, early_stopping = TRUE, batch_size = 50, verbose = FALSE)
    
    ### apply PHO
    deep_part <- get_layer(mod_wo_oz$model, "last_layer")
    intermediate_mod <- keras_model(mod_wo_oz$model$input, deep_part$output)
    
    newdata_processed <- deepregression:::prepare_newdata(
      mod_wo_oz$init_params$parsed_formulas_contents, relevant_data, 
      gamdata = mod_wo_oz$init_params$gamdata$data_trafos
    )
    
    zeta <- as.data.frame(intermediate_mod$predict(newdata_processed))
    X <- do.call("cbind", newdata_processed[1:p])
    H <- crossprod(X)
    s <- crossprod(X, as.matrix(as.matrix(zeta)))
    alpha <- solve(H, s)
    
    ### extract coefficients / partial effects

    plotdata <- NULL
    if(nonlin){
      plotdata <- lapply(list(mod_wo_oz, mod_w_oz), function(mod) mod %>% plot(only_data=TRUE, which_param=1))
      plotmod <- predict(true_mod, type="terms")
      gam_subtract <- gam(as.formula(paste0("V1 ~ 1 + ", paste(paste0("s(x", 1:p,")"), collapse=" + "))),
                          data = cbind(zeta, relevant_data))
      subtmod <- predict(gam_subtract, type="terms")
      coefs <- lapply(1:length(plotdata[[1]]), function(i){ 
        
        ret <- plotdata[[1]][[i]]
        ret$coef_oz <- plotdata[[2]][[i]]$coef
        ret$partial_effect_oz <- plotdata[[2]][[i]]$partial_effect
        ret$coef_pomo <- plotdata[[1]][[i]]$coef + alpha[(i-1)*9 + 1:9,] 
        ret$partial_effect_pomo <- plotdata[[1]][[i]]$partial_effect + 
          plotdata[[1]][[i]]$design_mat %*% (matrix(alpha[(i-1)*9 + 1:9,1]))
        ret$partial_effect_pomo2 <- plotdata[[1]][[i]]$partial_effect + 
          subtmod[,i,drop=F]
        ret$coef_gam <- coef(true_mod)[(i-1)*9 + 1:9]
        ret$gam_partial <- plotmod[,i,drop=F]
        return(ret)
        
      })
    }else{
      coefs <- cbind(
        with_ooz = c(unlist(coef(mod_w_oz, which_param = 1)[1:p])),
        without = c(unlist(coef(mod_wo_oz, which_param = 1)[1:p])),
        true_mod = coef(true_mod),
        pomodortho = c(unlist(coef(mod_wo_oz, which_param = 1)[1:p])) + alpha
      )
    }
    
    ### return
    
    return(list(coefs = coefs,
                histories = list(ooz = hist_ooz, woz = hist_woz)
    ))
    
  }, mc.cores = nr_cores)
  
  res_list[[i]] <- list(res = res, setting = settings[i,])

}

saveRDS(res_list, file="results_sim_ortho.RDS")


######################## Analysis #######################
if(FALSE){
  
  library(tidyverse)
  
  
  
  ### linear results
  
  res_lin <- do.call("rbind", lapply(res_list[settings$nonlin==0], 
                                     function(x) 
                                       cbind(do.call("rbind", lapply(x$res, "[[", "coefs")), x$setting)))
  
  res_lin %>% mutate(
    ONO = (with_ooz-true_mod)^2,
    unconstrained = (without-true_mod)^2,
    PHO = (V1-true_mod)^2
  ) %>% 
    select(ONO, unconstrained, PHO, n, p, p_nn) %>% 
    pivot_longer(ONO:PHO) %>% 
    mutate(name = fct_relevel(name, c("PHO","ONO","unconstrained"))) %>% 
    ggplot(aes(x = interaction(p,p_nn+p, sep = " / "), y=value, colour = name)) + 
    geom_boxplot() + 
    facet_grid(n~., labeller = labeller(n = label_both)) + 
    theme_bw() + ylab("RMSE") + xlab("p / q") + 
    theme(legend.title = element_blank(),
          text = element_text(size = 14)) + 
    scale_colour_manual(values = c("#999999", "#E69F00", "#56B4E9")) + 
    theme(legend.position="bottom")
  
  ggsave(width = 6, height = 4, filename = "results_linear.pdf")
  
  ### convergence
  res_conv <- do.call("rbind", lapply(res_list, function(rl)
    cbind(do.call("rbind", lapply(rl$res, function(rrl) 
      data.frame(ooz = length(rrl$histories$ooz$metrics$val_loss),
                 woz = length(rrl$histories$woz$metrics$val_loss)))), rl$setting)))
  
  colnames(res_conv)[5] <- "q"
  
  res_conv %>% mutate(
    diff_iter = ooz-woz
  ) %>% filter((nonlin==0 & n==100) | (nonlin==1 & n==1000)) %>% ggplot() + 
    geom_abline(intercept = 0, linetype = 2) +
    # geom_boxplot(aes(x = factor(p), y = diff_iter, fill = factor(q))) +
    geom_smooth(aes(x = p, y = diff_iter, colour = factor(q))) +
    facet_grid( ~ n, labeller = labeller(n = label_both)) +
    scale_colour_manual(values = c("#009E73", "#D55E00")) + 
    theme_bw() + 
    theme(legend.position="bottom",
          text = element_text(size = 14)) + 
    ylab("Additional iterations with constraint") + xlab("#Features p") + 
    scale_x_continuous(trans = "log10") +
    guides(colour=guide_legend(title="#Unstructured features q"))
  
  ggsave(width = 5, height = 4, filename = "convergence.pdf")
  
  ### non-linear results
  
  res_nonlin <- do.call("rbind", lapply(res_list[settings$nonlin==1], 
                                     function(x) 
                                       cbind(do.call("rbind", 
                                                     lapply(lapply(x$res, "[[", "coefs"),
                                                            function(z) do.call("rbind", lapply(1:length(z), function(sm)
                                                              data.frame(
                                                                effect = deepregression::extractvar(
                                                                  z[[sm]]$org_feature_name),
                                                                xvalue = z[[sm]]$value,
                                                                unconstrained = z[[sm]]$partial_effect[,1],
                                                                ONO = z[[sm]]$partial_effect_oz[,1],
                                                                PHO = z[[sm]]$partial_effect_pomo[,1],
                                                                PHOGAM = z[[sm]]$partial_effect_pomo2[,1],
                                                                GAM = z[[sm]]$gam_partial[,1],
                                                                truth = lotf[[sm]](z[[sm]]$value),
                                                                id = rnorm(1)
                                                              ))))), x$setting)))

  res_nonlin$effect[res_nonlin$effect == "x8"] <- "x0"
  res_nonlin$effect[res_nonlin$effect == "x10"] <- "x8"
  
    
  # res_nonlin %>% 
  #   pivot_longer(unconstrained:truth) %>% 
  #   filter(effect == "s(x1)") %>% 
  #   ggplot(aes(
  #   x = xvalue, y = value, color = name, group = factor(id)
  # )) + geom_point() + facet_wrap(~ n*p*p_nn)
  
  res_nonlin_rmse_per_x <- res_nonlin %>% mutate(
    ONO = (ONO-truth)^2,
    unconstrained = (unconstrained-truth)^2,
    PHO = (PHO-truth)^2,
    GAM = (GAM-truth)^2,
    PHOGAM = (PHOGAM-truth)^2
  ) %>% select(-truth,-nonlin,) %>% 
    pivot_longer(unconstrained:GAM) %>% 
    group_by(xvalue,id,n,p,p_nn,effect,name) %>% 
    summarise(value = sqrt(mean(value)))
  
  res_nonlin_rmse <- res_nonlin_rmse_per_x %>% 
    ungroup() %>% 
    group_by(id,n,p,p_nn,effect,name) %>% 
    summarise(value = sum(value))
  
  res_nonlin_rmse_per_s <- res_nonlin %>% mutate(
    ONO = (ONO-truth)^2,
    unconstrained = (unconstrained-truth)^2,
    PHO = (PHO-truth)^2,
    GAM = (GAM-truth)^2,
    PHOGAM = (PHOGAM-truth)^2
  ) %>% select(-truth,-nonlin,) %>% 
    pivot_longer(unconstrained:GAM) %>% 
    group_by(id,n,p,p_nn,effect,name) %>% 
    summarise(value = sqrt(mean(value)))
  
  res_nonlin_rmse_per_s %>% 
    ggplot(aes(
      x = effect, y = value, color = name #, group = factor(id)
    )) + geom_boxplot() + 
    facet_grid(n ~ ., scales = "free", labeller = labeller(n = label_both)) + 
    scale_colour_manual(values = c("#009E73", "#E69F00", "#999999", "#CC79A7", "#56B4E9")) +
    theme_bw() + 
    theme(legend.title = element_blank(),
          text = element_text(size = 14)) + 
    xlab("Feature") + ylab("RIMSE") + 
    theme(legend.position="bottom")
  
  ggsave(width = 6, height = 4, filename = "results_nonlinear.pdf")
  
  res_nonlin_agg <- res_nonlin %>% filter(p==10, p_nn==0, n==1000) %>% 
    filter(!effect%in%c("x0","x7","x8","x9")) %>%
    pivot_longer(unconstrained:truth) %>% 
    arrange(name, effect, id, xvalue) %>% 
    group_by(effect, id, name) %>% 
    mutate(value = ifelse(name=="truth", value-mean(value), value)) %>% 
    ungroup() %>% 
    group_by(name, effect, xvalue) %>% 
    summarise(
      meanval = mean(value),
      uppval = quantile(value, probs = 0.95),
      lowval = quantile(value, probs = 0.05)
    ) %>% 
    ungroup()
  
  ggplot(res_nonlin_agg, aes(x = xvalue, y = meanval, colour = name, group = name)) + 
    geom_line(linewidth = 1.5) + 
    # geom_ribbon(aes(ymin = lowval, ymax = uppval, fill = name), alpha = 0.1) + 
    theme_bw() + facet_wrap(~effect, scales="free", ncol=3) + 
    scale_colour_manual(values = c("#009E73", "#E69F00", "#999999", "#CC79A7", "#FF0000", "#56B4E9")) +
    scale_fill_manual(values = c("#009E73", "#E69F00", "#999999", "#CC79A7", "#FF0000", "#56B4E9")) +
    theme(legend.title = element_blank(),
          text = element_text(size = 14)) + xlab("Feature value") + ylab("Partial effect") + 
    guides(colour = guide_legend(override.aes = list(alpha=1, size=1.1),
                                 nrow=1,byrow=TRUE)) + 
    theme(legend.position="bottom") 
    
  ggsave(width = 6.5, height = 4, filename = "results_splines.pdf")
  
}
