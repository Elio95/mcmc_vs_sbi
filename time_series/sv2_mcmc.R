###############################################################################
# SV2 MCMC using stochvol package
###############################################################################

library(stochvol)
library(readr)
library(dplyr)
library(arrow)

# 1) Load and prepare S&P 500 data
df <- read.csv("SP500.csv", header = TRUE, stringsAsFactors = FALSE)

# Check column names - expect Date and Close
head(df)

# Remove missing rows
df <- df %>% filter(!is.na(Close))

# Compute log-returns
df <- df %>%
  mutate(
    logreturn = log(Close) - log(lag(Close))
  ) %>%
  filter(!is.na(logreturn))

# Extract returns as numeric vector
returns <- df$logreturn

# 2) Build design matrix (intercept + lagged return)
lag1 <- c(NA, returns[-length(returns)])
design_data <- data.frame(
  intercept = 1,
  lagret = lag1
)

# Remove first row (NA in lag)
design_data <- design_data[-1, ]
my_returns <- returns[-1]

cat("Data dimensions:\n")
cat("  Returns:", length(my_returns), "\n")
cat("  Design matrix:", nrow(design_data), "x", ncol(design_data), "\n\n")

# 3) Run MCMC with stochvol
set.seed(123)

ndraw <- 50000
burnin <- 5000

cat("Starting MCMC inference...\n")
cat("  Draws:", ndraw, "\n")
cat("  Burn-in:", burnin, "\n")
cat("  Thinning: 5\n\n")

start_time <- Sys.time()

res <- svsample(
  y = my_returns,
  designmatrix = as.matrix(design_data),
  draws = ndraw,
  burnin = burnin,
  priormu = c(0, 10),          # mu ~ N(0, 10^2)
  priorphi = c(5, 1.5),         # Beta(5, 1.5) for phi
  priorsigma = 0.2,             # log(sigma) ~ N(log(0.3), 0.2^2)
  priorbeta = c(0, 1),          # beta ~ N(0, 1^2)
  quiet = FALSE,
  thin = 5
)

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("\n")
cat("="*80, "\n", sep="")
cat("MCMC COMPLETE\n")
cat("="*80, "\n", sep="")
cat("Runtime:", round(runtime, 2), "seconds\n\n")

# 4) Summary and diagnostics
print(summary(res))
# Convergence diagnostics
plot(res)

# 5) Extract and save posterior draws
library(coda)

# Convert to mcmc objects
mcmc_para <- as.mcmc(res$para)
mcmc_beta <- as.mcmc(res$beta)
mcmc_latent <- as.mcmc(res$latent)

# Convert to data frames
para_draws_df <- as.data.frame(mcmc_para)
beta_draws_df <- as.data.frame(mcmc_beta)
latent_draws_df <- as.data.frame(mcmc_latent)

# Save to Feather files for Python interoperability
write_feather(para_draws_df, "posterior_params.feather")
write_feather(beta_draws_df, "posterior_beta.feather")
write_feather(latent_draws_df, "posterior_latent.feather")

cat("\nPosterior samples saved:\n")
cat("  posterior_params.feather  -", nrow(para_draws_df), "draws\n")
cat("  posterior_beta.feather    -", nrow(beta_draws_df), "draws\n")
cat("  posterior_latent.feather  -", nrow(latent_draws_df), "draws\n")

# 6) Posterior summary statistics
summary_stats <- data.frame(
  parameter = c("mu", "phi", "sigma", "beta0", "beta1"),
  mean = c(
    mean(para_draws_df$mu),
    mean(para_draws_df$phi),
    mean(para_draws_df$sigma),
    mean(beta_draws_df$beta_0),
    mean(beta_draws_df$beta_1)
  ),
  sd = c(
    sd(para_draws_df$mu),
    sd(para_draws_df$phi),
    sd(para_draws_df$sigma),
    sd(beta_draws_df$beta_0),
    sd(beta_draws_df$beta_1)
  )
)

print(summary_stats)
write.csv(summary_stats, "mcmc_summary.csv", row.names = FALSE)
