library(deepregression)
library(mgcv)
library(safareg)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)

data <- readRDS("train.RDS")
testrun <- TRUE

if(testrun){
  set.seed(123)
  data <- data %>% sample_n(1e7)
}
  
deep_net <- function(x){
  
  x %>% layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>% 
    layer_dense(units = 100, activation = "relu") %>% 
    layer_dense(units = 100, activation = "relu") %>% 
    layer_dense(units = 1, name = "last_layer")
  
}

mod <- deepregression(y = data$ArrDelay,
                      list_of_formulas = list(
                        ~ 1 + s(Year, bs="bs") + s(Month, bs="bs") + s(DayofMonth, bs="bs") + 
                          fac(DayOfWeek) + 
                          s(CRSDepTime, bs="cp") + s(CRSArrTime, bs="cp") + 
                          fac(UniqueCarrier) + fac(Origin) + fac(Dest) + 
                          s(Distance, bs="bs") + 
                          fac(Route) + dnet(Year, Month, DayofMonth, CRSDepTime, CRSArrTime, Distance),
                        ~ 1
                      ),
                      list_of_deep_models = list(dnet = deep_net),
                      additional_processors = list(fac = fac_processor),
                      orthog_options = orthog_control(orthogonalize = FALSE),
                      data = data
)

hist <- mod %>% fit(epochs = 120,
                    validation_split = 0.5,
                    batch_size = 2000,
                    early_stopping = TRUE,
                    patience = 5,
                    verbose = TRUE
)

plot(hist)

save_model_weights_hdf5(mod$model, filepath=paste0("mod_flights.hdf5"))

coefs <- mod %>% coef()
saveRDS(coefs, file="coef_flights.RDS")
pred <- mod %>% predict(batch_size = 1000L)
saveRDS(pred, file="prediction_flights.RDS")

deep_part <- get_layer(mod$model, "last_layer")
intermediate_mod <- keras_model(mod$model$input, deep_part$output)

newdata_processed <- deepregression:::prepare_newdata(
  mod$init_params$parsed_formulas_contents, data, 
  gamdata = mod$init_params$gamdata$data_trafos
)

deep_part <- intermediate_mod$predict(newdata_processed)[,1]
saveRDS(deep_part, "deeppart_flights.RDS")

plotdata <- mod %>% plot(only_data=TRUE, which_param=1)
saveRDS(plotdata, file="plotdata_flights.RDS")

X <- do.call("cbind", newdata_processed)
saveRDS(X, file="designmat_flights.RDS")

######## PHO #########

rm(mod, coefs, pred, intermediate_mod, newdata_processed, X)
gc(); gc()

data$zeta <- deep_part

gam_subtract <- bam(deep_part ~ 1 + s(Year, bs="bs") + s(Month, bs="bs") + s(DayofMonth, bs="bs") + 
                      DayOfWeek + 
                      s(CRSDepTime, bs="cp") + s(CRSArrTime, bs="cp") + 
                      UniqueCarrier + Origin + Dest + 
                      s(Distance, bs="bs") + 
                      Route,
                    data = data, discrete = 20, nthreads = 20)
saveRDS(gam_subtract, file="gam_flights.RDS")

subtmod <- predict(gam_subtract, type="terms")
saveRDS(subtmod, file="subtmod_flights.RDS")


# nr_batches <- 1000
# 
# idx <- 1:nrow(X)
# f <- as.factor((seq_along(idx) - 1) %% nr_batches)
# # split the vector
# split_idx <- split(idx, f)
# 
# Hs <- Reduce("+", lapply(split_idx, function(id){
# 
#   crossprod(X[id,])
# 
# }))
# ss <- Reduce("+", lapply(split_idx, function(id){
# 
#   crossprod(X[id,], as.matrix(as.matrix(deep_part[id])))
# 
# }))
# 
# alpha <- solve(Hs, ss)
# 
plotdata_pho <- lapply(1:length(plotdata), function(i) plotdata[[i]]$partial_effect + 
                         subtmod[,(6:11)[i],drop=F])
saveRDS(plotdata_pho, file="plotdata_pho_flights.RDS")

plotdata <- data.frame(x = c(unlist(lapply(plotdata, "[[", "value"))),
                       sx = c(unlist(lapply(plotdata, function(x) x$partial_effect[,1]))),
                       sx_pho = c(unlist(lapply(plotdata_pho, function(x) x[,1]))),
                       featname = rep(c("s(Year)", "s(Month)", "s(DayofMonth)",
                                        "s(CRSDepTime)", "s(CRSArrTime)", 
                                        "s(Distance)"), each=1e7))

saveRDS(plotdata, file="plotdata.RDS")
plotdata <- plotdata %>% sample_n(1e6)
saveRDS(plotdata, file="plotdata_subset.RDS")

gc()

gg <- ggplot(data = plotdata %>% 
               pivot_longer(sx:sx_pho) %>% 
               mutate(name = factor(name, levels = c("sx", "sx_pho"), labels = c("SSN", "PHO")),
                      featname = factor(featname, levels = unique(plotdata$featname)[c(1,5,4,2,3,6)],
                                        labels = c("year", 
                                                   "month",
                                                   "day of month", 
                                                   "distance", 
                                                   "sched. departure time (min + 0:00)", 
                                                   "sched. arrival time (min + 0:00)"))),
             aes(x = x, y = value, colour = name)) + geom_line(linewidth = 1.3) + 
  facet_wrap(~featname, scales = "free", ncol=2) + 
  theme_bw() + xlab("Value") + ylab("Partial Effect") + 
  scale_colour_manual(values = c("#009E73", "#CC79A7")) + 
  theme(text = element_text(size = 14),
        legend.title = element_blank(),
        legend.position = "bottom")

gg

ggsave(filename = "partial_effects_flight.pdf", width = 6, height = 6)
