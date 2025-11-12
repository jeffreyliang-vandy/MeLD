#!/usr/bin/env Rscript

# Load necessary libraries
suppressPackageStartupMessages({
    library(tidyverse)
    library(rms)
    library(survival)
})

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: script.R <data_path> <output_csv>")
}

data_path <- args[1]
output_csv <- args[2]

# Load data
suppressMessages(surv_dt <- read_csv(data_path))

surv_dt = filter(surv_dt,male_y!='8')
surv_dt['center'] = relevel(as.factor(surv_dt$center),ref = "1")

# Set datadist for rms package
dd <- datadist(surv_dt)
dd$limits[c(1,3),'age'] = c(20,30)
dd$limits[c(1,3),'enrol_d'] = c(2010,2020)
dd$limits[c(1,3),'cd4_v'] = c(100,350)
dd$limits[c(1),'center'] = 1
options(datadist = "dd", digits = 3)

# Run Cox proportional hazards model
suppressWarnings({
df_cox <- cph(
  Surv(time = time, event = event) ~ rcs(enrol_d, 4) + rcs(age, 3) + rcs(cd4_v, 4) + 
                                        male_y + strat(center),
  data = surv_dt
)
})

# Extract coefficients
suppressWarnings({
coefficients <- as.data.frame(summary(df_cox))
coefficients <- tibble::rownames_to_column(coefficients, var = "Variable")
})

# Save coefficients to CSV
write.table(coefficients, file = output_csv, sep = ",", row.names = TRUE, col.names = !file.exists(output_csv), append = TRUE)

cat("Coefficients saved to", output_csv, "\n")
