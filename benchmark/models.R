############################# Loading libraries #############################
library(devtools)
library(deepregression)

### architectures considered

get_deep_mod <- function(architecture = c("a", "b", "c", "d")){
  
  architecture <- match.arg(architecture)
  deep_mod <- switch(architecture,
                     a = function(x) x %>% 
                       layer_dense(units = 200, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     b = function(x) x %>% 
                       layer_dense(units = 200, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 200, activation = "relu") %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     c = function(x) x %>% 
                       layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear"),
                     d = function(x) x %>% 
                       layer_dense(units = 20, activation = "relu", use_bias = FALSE) %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 20, activation = "relu") %>%
                       layer_dropout(rate = 0.1) %>% 
                       layer_dense(units = 1, activation = "linear")
  )
  
  return(deep_mod)
}

# optimizer_alig <- reticulate::import_from_path("codes", paste0(getwd(), "/.."))

############################# Generic Normal (Deep) Regression ################################
dr <- function(formla, trainX, trainY, testX,
               deep_mod_list = NULL,
               additional_processors = NULL,
               maxEpochs = 1000,
               patience = 50, 
               optimizer = optimizer_adam(),
               verbose = FALSE,
               oz_option = orthog_control()
               ){
  
  family <- "normal"
    
  if(length(unique(trainY))<=5) family = "multinomial"

  args <- list(y = trainY, 
               list_of_formulas = list(as.formula(formla), ~1),
               list_of_deep_models = deep_mod_list,
               additional_processors = additional_processors,
               data = trainX, 
               family = family,
               optimizer = optimizer,
               orthog_options = oz_option)

  mod <- do.call("deepregression", args)
  
  mod %>% fit(epochs = maxEpochs, early_stopping = TRUE, 
              patience = patience, verbose = verbose)
  
  pred <- mod %>% predict(testX)

  return(pred)
  
}


############################# MLP #############################
mlp <- function(architecture,
                trainX, trainY, testX,
                maxEpochs = 1000,
                patience = 50,
                ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vs <- colnames(trainX)
  
  form <- paste0("~ 1", 
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  pred <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX = trainX, trainY = trainY, testX = testX,
             ...)
  
  return(pred)
  
  
}
############################# GAM #############################
gamr <- function(trainX, trainY, testX, ...){
  
  
  Vs <- colnames(trainX)
  
  form <- paste0("~ 1 + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "))
  
  pred <- dr(form, trainX, trainY, testX, ...)
  
  return(pred)
  
  
}

############################# SSN #############################
pho <- function(architecture, trainX, trainY, testX, ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vs <- colnames(trainX)
  
  form <- paste0("~ 1 + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "),
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  pred <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = FALSE),
             ...)
  
  return(pred)
  
  
}

############################# ONO #############################
ono <- function(architecture, trainX, trainY, testX, ...){
  
  deep_mod <- get_deep_mod(architecture)
  
  Vs <- colnames(trainX)
  
  form <- paste0("~ 1 + ",
                 paste(paste0("s(", Vs, ")"), collapse=" + "),
                 " + deep_mod(",
                 paste(Vs, collapse=", "), ")")
  
  pred <- dr(form, deep_mod_list = list(deep_mod = deep_mod),
             trainX, trainY, testX, 
             oz_option = orthog_control(orthogonalize = TRUE), 
             ...)
  
  return(pred)
  
  
}
