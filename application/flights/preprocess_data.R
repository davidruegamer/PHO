library(readr)
library(dplyr)
library(data.table)

data <- rbindlist(lapply(list.files("data/", 
                                    pattern = ".csv", 
                                    full.names = T), 
                         function(x){
                           
                           data <- read_csv(x)
                           data <- data %>% select(Year, Month, DayofMonth, # -> splines 
                                                   DayOfWeek, # -> factor 
                                                   CRSDepTime, CRSArrTime, # -> cyclic splines 
                                                   UniqueCarrier, # -> factor
                                                   ArrDelay, # -> response 
                                                   Origin, Dest, # -> factor 
                                                   Distance, # splines
                                                   Cancelled, Diverted)
                           data <- data %>% filter(Cancelled == 0 & Diverted == 0)
                           data <- data %>% select(-Cancelled, -Diverted)
                           return(data)
                           
                         }))

total_years <- data[, uniqueN(Year)]

carriers <- data[, .(num_years = uniqueN(Year)), by = UniqueCarrier][num_years == total_years, UniqueCarrier]
dests <- data[, .(num_years = uniqueN(Year)), by = Dest][num_years == total_years, Dest]
origins <- data[, .(num_years = uniqueN(Year)), by = Origin][num_years == total_years, Origin]

data <- data[UniqueCarrier %in% carriers & Dest %in% dests & Origin %in% origins]

data <- na.omit(data)

data <- data %>% 
  mutate(
    Route = interaction(Origin, Dest, sep = "-")
  )
data[, UniqueCarrier := as.factor(UniqueCarrier)]
data[, Origin := as.factor(Origin)]
data[, Dest := as.factor(Dest)]
data[, DayOfWeek := as.factor(DayOfWeek)]

recode_time <- function(num) {
  hundreds <- num %/% 100
  hundreds * 60 + num %% 100
}

data[, CRSDepTime := recode_time(CRSDepTime)]
data[, CRSArrTime := recode_time(CRSArrTime)]

saveRDS(data, file="flights.RDS")

### Train/Test split

set.seed(123)

train_prop <- 0.5
n <- nrow(data)

train_indices <- sample(n, round(n*train_prop))
test_indices <- setdiff(1:n, train_indices)

# Create the training set
train <- data[-test_indices,]
saveRDS(train, file="train.RDS")

# Create the test set
test <- data[-train_indices,]
saveRDS(test, file="test.RDS")