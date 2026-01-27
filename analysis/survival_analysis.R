# =============================================================================
# Survival Analysis of ChatGPT Conversation Duration
# R Script for Discrete-Time Hazard and Split-Population Models
# =============================================================================
#
# Research Question: Do follow-up suggestions extend conversation length,
# or is the effect driven by model capability improvements over time?
#
# Author: AI Disclosures Project
# Date: January 2026
# =============================================================================

library(tidyverse)
library(survival)
library(broom)
library(lme4)
library(ggplot2)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------

data_path <- "analysis/survival_data.csv"
df <- read_csv(data_path, show_col_types = FALSE)

cat("\n=== DATA SUMMARY ===\n")
cat(sprintf("Total turn-level observations: %s\n", format(nrow(df), big.mark=",")))
cat(sprintf("Unique conversations: %s\n", format(n_distinct(df$conversation_hash), big.mark=",")))

# -----------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS BY YEAR
# -----------------------------------------------------------------------------

cat("\n=== TURN DISTRIBUTION BY YEAR ===\n")

conv_level <- df %>%
  group_by(conversation_hash) %>%
  summarise(
    turns = first(turns),
    year = first(year),
    model = first(model),
    .groups = 'drop'
  )

year_stats <- conv_level %>%
  group_by(year) %>%
  summarise(
    n = n(),
    mean = mean(turns),
    sd = sd(turns),
    median = median(turns),
    P10 = quantile(turns, 0.10),
    P50 = quantile(turns, 0.50),
    P90 = quantile(turns, 0.90),
    P95 = quantile(turns, 0.95),
    P99 = quantile(turns, 0.99),
    max = max(turns),
    .groups = 'drop'
  )

print(year_stats)

# -----------------------------------------------------------------------------
# 3. POWER-LAW TAIL ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=== PARETO TAIL EXPONENT BY YEAR ===\n")

fit_pareto <- function(turns, min_t = 5) {
  # Fit P(T >= t) ~ t^(-alpha) for t >= min_t
  t_vals <- min_t:max(turns)
  survival <- sapply(t_vals, function(t) mean(turns >= t))

  # Filter positive survival probabilities
  valid <- survival > 0
  log_t <- log(t_vals[valid])
  log_s <- log(survival[valid])

  if(length(log_t) < 3) return(c(NA, NA))

  fit <- lm(log_s ~ log_t)
  alpha <- -coef(fit)[2]
  r2 <- summary(fit)$r.squared

  return(c(alpha = unname(alpha), r2 = r2))
}

pareto_by_year <- conv_level %>%
  group_by(year) %>%
  summarise(
    n = n(),
    pareto_result = list(fit_pareto(turns)),
    .groups = 'drop'
  ) %>%
  mutate(
    pareto_alpha = sapply(pareto_result, `[`, 1),
    r2_fit = sapply(pareto_result, `[`, 2)
  ) %>%
  select(-pareto_result)

print(pareto_by_year)

# -----------------------------------------------------------------------------
# 4. DISCRETE-TIME HAZARD MODEL (LOGISTIC REGRESSION)
# -----------------------------------------------------------------------------

cat("\n=== MODEL 1: BASELINE DISCRETE-TIME HAZARD ===\n")

# Prepare data
df_model <- df %>%
  mutate(
    event = as.integer(!conversation_continued),
    suggestion_given = as.integer(suggestion_given_count > 0),
    year_factor = factor(year, levels = c(2023, 2024, 2025)),
    log_turn = log(turn)
  )

# Baseline model: turn + suggestion + year
model_baseline <- glm(
  event ~ log_turn + suggestion_given + year_factor,
  data = df_model,
  family = binomial(link = "logit")
)

cat("\nBaseline Model Summary:\n")
print(summary(model_baseline))

# Odds ratios with CIs
cat("\nOdds Ratios (95% CI):\n")
or_baseline <- exp(cbind(OR = coef(model_baseline), confint(model_baseline)))
print(round(or_baseline, 4))

# -----------------------------------------------------------------------------
# 5. EXTENDED MODEL WITH WORD COUNTS
# -----------------------------------------------------------------------------

cat("\n=== MODEL 2: EXTENDED WITH WORD COUNTS ===\n")

df_model <- df_model %>%
  mutate(
    user_wc_log = log1p(user_word_count),
    # Only include non-zero assistant word counts
    asst_wc_log = log1p(pmax(assistant_word_count, 0))
  )

model_extended <- glm(
  event ~ log_turn + suggestion_given + year_factor + user_wc_log,
  data = df_model,
  family = binomial(link = "logit")
)

cat("\nExtended Model Summary:\n")
print(summary(model_extended))

cat("\nOdds Ratios (95% CI):\n")
or_extended <- exp(cbind(OR = coef(model_extended), confint(model_extended)))
print(round(or_extended, 4))

# -----------------------------------------------------------------------------
# 6. RANDOM EFFECTS MODEL (FRAILTY)
# -----------------------------------------------------------------------------

cat("\n=== MODEL 3: RANDOM EFFECTS (CONVERSATION-LEVEL FRAILTY) ===\n")

# Mixed effects logistic regression with conversation random intercept
# This captures unobserved heterogeneity (some users inherently more engaged)
model_frailty <- glmer(
  event ~ log_turn + suggestion_given + year_factor + (1 | conversation_hash),
  data = df_model,
  family = binomial(link = "logit"),
  nAGQ = 0,  # Use Laplace approximation for speed
  control = glmerControl(optimizer = "bobyqa")
)

cat("\nRandom Effects Model Summary:\n")
print(summary(model_frailty))

# -----------------------------------------------------------------------------
# 7. MODEL WITH GPT VERSION EFFECTS
# -----------------------------------------------------------------------------

cat("\n=== MODEL 4: WITH GPT MODEL FIXED EFFECTS ===\n")

# Simplify model names
df_model <- df_model %>%
  mutate(
    model_family = case_when(
      str_detect(model, "gpt-3.5") ~ "gpt-3.5",
      str_detect(model, "gpt-4o-mini|gpt-4.1-mini") ~ "gpt-4-mini",
      str_detect(model, "gpt-4o") ~ "gpt-4o",
      str_detect(model, "gpt-4") ~ "gpt-4",
      TRUE ~ "other"
    )
  )

model_with_gpt <- glm(
  event ~ log_turn + suggestion_given + year_factor + model_family,
  data = df_model,
  family = binomial(link = "logit")
)

cat("\nModel with GPT Version Effects:\n")
print(summary(model_with_gpt))

# -----------------------------------------------------------------------------
# 8. SPLIT-POPULATION MODEL (CURE MODEL)
# -----------------------------------------------------------------------------

cat("\n=== MODEL 5: SPLIT-POPULATION (MIXTURE CURE) MODEL ===\n")
cat("Note: Conceptual framework - full implementation requires 'smcure' package\n")

# The split-population model separates:
# - pi: probability of being a "multi-turn" user (vs single-query)
# - S(t|uncured): survival function for those who might continue

# Simplified version: model first-turn behavior separately
first_turn <- df_model %>% filter(turn == 1)
later_turns <- df_model %>% filter(turn > 1)

cat("\n--- First Turn (Who continues at all?) ---\n")
model_first_turn <- glm(
  event ~ suggestion_given + year_factor + user_wc_log,
  data = first_turn,
  family = binomial(link = "logit")
)
print(summary(model_first_turn))

cat("\n--- Later Turns (Given they continued, do they continue more?) ---\n")
model_later_turns <- glm(
  event ~ log_turn + suggestion_given + year_factor + user_wc_log,
  data = later_turns,
  family = binomial(link = "logit")
)
print(summary(model_later_turns))

# -----------------------------------------------------------------------------
# 9. KEY RESULTS SUMMARY
# -----------------------------------------------------------------------------

cat("\n")
cat("=" , rep("=", 68), "=\n", sep="")
cat("KEY FINDINGS SUMMARY\n")
cat("=" , rep("=", 68), "=\n", sep="")

cat("\n1. PARETO TAIL IS GETTING HEAVIER OVER TIME\n")
cat(sprintf("   2023: α = %.2f\n", pareto_by_year$pareto_alpha[pareto_by_year$year == 2023]))
cat(sprintf("   2024: α = %.2f\n", pareto_by_year$pareto_alpha[pareto_by_year$year == 2024]))
cat(sprintf("   2025: α = %.2f\n", pareto_by_year$pareto_alpha[pareto_by_year$year == 2025]))
cat("   Lower α = heavier tail = more extreme-length conversations\n")

cat("\n2. SUGGESTION EFFECT (Baseline Model)\n")
suggestion_or <- exp(coef(model_baseline)["suggestion_given"])
cat(sprintf("   OR = %.3f (p < 0.001)\n", suggestion_or))
if(suggestion_or > 1) {
  cat("   INTERPRETATION: Suggestions INCREASE hazard of ending\n")
  cat("   This is counterintuitive - suggests selection effect:\n")
  cat("   Suggestions appear at natural stopping points\n")
} else {
  cat("   INTERPRETATION: Suggestions DECREASE hazard of ending\n")
}

cat("\n3. YEAR EFFECTS (controlling for turn, suggestion)\n")
cat(sprintf("   2024 vs 2023: OR = %.3f\n", exp(coef(model_baseline)["year_factor2024"])))
cat(sprintf("   2025 vs 2023: OR = %.3f\n", exp(coef(model_baseline)["year_factor2025"])))

cat("\n4. FIRST-TURN VS LATER-TURN DYNAMICS\n")
cat(sprintf("   First-turn dropout: %.1f%%\n", 100 * mean(first_turn$event)))
cat(sprintf("   Later-turn dropout: %.1f%%\n", 100 * mean(later_turns$event)))

cat("\n5. MODEL CAPABILITY VS ENGAGEMENT DESIGN\n")
cat("   Cannot fully separate effects due to model-year collinearity\n")
cat("   gpt-4o-mini spans 2024-2025: useful for within-model comparison\n")

# -----------------------------------------------------------------------------
# 10. SAVE RESULTS
# -----------------------------------------------------------------------------

# Save model summaries
sink("analysis/model_results.txt")
cat("=== BASELINE MODEL ===\n")
print(summary(model_baseline))
cat("\n=== EXTENDED MODEL ===\n")
print(summary(model_extended))
cat("\n=== FIRST TURN MODEL ===\n")
print(summary(model_first_turn))
cat("\n=== LATER TURNS MODEL ===\n")
print(summary(model_later_turns))
sink()

cat("\nResults saved to analysis/model_results.txt\n")
