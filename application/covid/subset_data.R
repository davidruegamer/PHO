# Download data from:
# https://storage.googleapis.com/covid19-open-data/v3/aggregated.csv.gz

# Read and subset data
library(data.table)
library(dplyr)

df <- fread(file="aggregated.csv")
df <- df %>% select(date, country_code, aggregation_level,
                    new_confirmed, population, 
              latitude, longitude,
              average_temperature_celsius, relative_humidity) %>% 
  filter(aggregation_level == 2, country_code == "US") %>% 
  filter(!is.na(latitude) & !is.na(longitude) & 
           !is.na(new_confirmed) & 
           !is.na(average_temperature_celsius) & 
           !is.na(relative_humidity)) %>% 
  mutate(date = as.numeric(as.Date(date) - as.Date("2020-01-01")),
         prevalence = new_confirmed / population)

fwrite(df, "aggregated_us_lev2_subset.csv")
