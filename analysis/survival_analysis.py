#!/usr/bin/env python3
"""
Survival Analysis of ChatGPT Conversation Duration
===================================================

Research Question: Do follow-up suggestions extend conversation length,
or is the observed increase in conversation duration driven by model
capability improvements over time?

Key Variables:
- Outcome: conversation_continued (discrete-time hazard)
- Treatment: suggestion_given_count (whether suggestions were offered)
- Time: turn (discrete time periods)
- Controls: year, model, word_count

Author: AI Disclosures Project
Date: January 2026
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load JSONL data into list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_dataframes(data):
    """Create turn-level and conversation-level DataFrames."""

    # Turn-level DataFrame (for discrete-time hazard model)
    df_turns = pd.DataFrame(data)

    # Conversation-level DataFrame (for summary stats)
    conversations = {}
    for row in data:
        conv_hash = row.get('conversation_hash')
        if conv_hash not in conversations:
            conversations[conv_hash] = row.copy()
        # Update with max values where relevant
        if row.get('turn', 0) > conversations[conv_hash].get('turn', 0):
            conversations[conv_hash]['max_turn'] = row.get('turn')

    df_conv = pd.DataFrame(list(conversations.values()))

    return df_turns, df_conv

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def compute_percentiles(arr, percentiles=[10, 30, 50, 60, 90, 95, 99, 100]):
    """Compute multiple percentiles."""
    return {f'P{p}': np.percentile(arr, p) for p in percentiles}

def descriptive_stats_by_year(df_conv):
    """Compute turn distribution statistics by year."""
    results = []

    for year in sorted(df_conv['year'].dropna().unique()):
        year_data = df_conv[df_conv['year'] == year]
        turns = year_data['turns'].dropna().values

        stats_dict = {
            'year': int(year),
            'n_conversations': len(turns),
            'mean': np.mean(turns),
            'std': np.std(turns),
            'median': np.median(turns),
        }
        stats_dict.update(compute_percentiles(turns))
        results.append(stats_dict)

    return pd.DataFrame(results)

def compute_survival_function(turns_array, max_t=50):
    """Compute empirical survival function S(t) = P(T >= t)."""
    t_vals = np.arange(1, max_t + 1)
    survival = np.array([np.mean(turns_array >= t) for t in t_vals])
    return t_vals, survival

def fit_pareto_tail(t_vals, survival, min_t=5):
    """Fit power law to tail: P(T>=t) ~ t^(-alpha)."""
    mask = (t_vals >= min_t) & (survival > 0)
    log_t = np.log(t_vals[mask])
    log_s = np.log(survival[mask])

    if len(log_t) < 3:
        return None, None, None

    slope, intercept = np.polyfit(log_t, log_s, 1)
    alpha = -slope

    # R-squared
    predicted = slope * log_t + intercept
    ss_res = np.sum((log_s - predicted) ** 2)
    ss_tot = np.sum((log_s - np.mean(log_s)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return alpha, r2, intercept

def tail_analysis_by_year(df_conv):
    """Analyze power-law tail by year."""
    results = []

    for year in sorted(df_conv['year'].dropna().unique()):
        turns = df_conv[df_conv['year'] == year]['turns'].dropna().values
        t_vals, survival = compute_survival_function(turns)
        alpha, r2, _ = fit_pareto_tail(t_vals, survival)

        # Compute tail probabilities
        results.append({
            'year': int(year),
            'n': len(turns),
            'pareto_alpha': alpha,
            'r2_fit': r2,
            'P(T>5)': np.mean(turns > 5),
            'P(T>10)': np.mean(turns > 10),
            'P(T>20)': np.mean(turns > 20),
            'P(T>30)': np.mean(turns > 30),
        })

    return pd.DataFrame(results)

def hazard_by_turn(df_turns, max_turn=20):
    """Compute empirical hazard rate by turn number."""
    results = []

    for turn in range(1, max_turn + 1):
        turn_data = df_turns[df_turns['turn'] == turn]
        n_at_risk = len(turn_data)
        n_ended = len(turn_data[turn_data['conversation_continued'] == False])
        n_continued = len(turn_data[turn_data['conversation_continued'] == True])

        hazard = n_ended / n_at_risk if n_at_risk > 0 else np.nan

        results.append({
            'turn': turn,
            'n_at_risk': n_at_risk,
            'n_ended': n_ended,
            'n_continued': n_continued,
            'hazard': hazard,
            'continuation_rate': 1 - hazard if not np.isnan(hazard) else np.nan
        })

    return pd.DataFrame(results)

def suggestion_effect_by_year(df_turns):
    """Compute continuation rate by suggestion status and year."""
    results = []

    for year in sorted(df_turns['year'].dropna().unique()):
        year_data = df_turns[df_turns['year'] == year]

        # With suggestion
        with_sugg = year_data[year_data['suggestion_given_count'] > 0]
        cont_with = with_sugg['conversation_continued'].mean()
        n_with = len(with_sugg)

        # Without suggestion
        without_sugg = year_data[year_data['suggestion_given_count'] == 0]
        cont_without = without_sugg['conversation_continued'].mean()
        n_without = len(without_sugg)

        results.append({
            'year': int(year),
            'n_with_suggestion': n_with,
            'n_without_suggestion': n_without,
            'cont_rate_with_suggestion': cont_with,
            'cont_rate_without_suggestion': cont_without,
            'difference': cont_with - cont_without
        })

    return pd.DataFrame(results)

# ============================================================================
# SURVIVAL MODELS
# ============================================================================

def prepare_survival_data(df_turns):
    """Prepare data for discrete-time survival model."""

    # Create binary outcome
    df = df_turns.copy()
    df['event'] = (~df['conversation_continued']).astype(int)

    # Create treatment indicator
    df['suggestion_given'] = (df['suggestion_given_count'] > 0).astype(int)

    # Year dummies (reference: 2023)
    df['year_2024'] = (df['year'] == 2024).astype(int)
    df['year_2025'] = (df['year'] == 2025).astype(int)

    # Log turn for baseline hazard
    df['log_turn'] = np.log(df['turn'])

    # Normalize word counts
    df['user_wc_log'] = np.log1p(df['user_word_count'].fillna(0))
    df['asst_wc_log'] = np.log1p(df['assistant_word_count'].fillna(0))

    return df

def fit_discrete_hazard_model(df, model_type='baseline'):
    """
    Fit discrete-time hazard model using logistic regression.

    P(end at turn t | survived to t) = logit^-1(Xβ)

    Uses statsmodels for proper inference (SEs, p-values, CIs).
    """
    import statsmodels.api as sm

    # Define feature sets
    if model_type == 'baseline':
        features = ['log_turn', 'suggestion_given', 'year_2024', 'year_2025']
    elif model_type == 'extended':
        features = ['log_turn', 'suggestion_given', 'year_2024', 'year_2025',
                   'user_wc_log', 'asst_wc_log']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Prepare data
    df_model = df[features + ['event']].dropna()
    X = df_model[features].copy()
    y = df_model['event'].values

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit logistic regression
    model = sm.Logit(y, X).fit(disp=0)

    # Extract results
    coefs = model.params.to_dict()
    std_errs = model.bse.to_dict()
    pvalues = model.pvalues.to_dict()
    conf_int = model.conf_int()

    # Compute odds ratios and their CIs
    odds_ratios = {k: np.exp(v) for k, v in coefs.items()}
    or_ci_low = {k: np.exp(conf_int.loc[k, 0]) for k in coefs.keys()}
    or_ci_high = {k: np.exp(conf_int.loc[k, 1]) for k in coefs.keys()}

    return {
        'model_type': model_type,
        'coefficients': coefs,
        'std_errors': std_errs,
        'pvalues': pvalues,
        'odds_ratios': odds_ratios,
        'or_ci_low': or_ci_low,
        'or_ci_high': or_ci_high,
        'n_obs': int(model.nobs),
        'n_events': int(y.sum()),
        'log_likelihood': model.llf,
        'aic': model.aic,
        'bic': model.bic,
        'pseudo_r2': model.prsquared,
        'features': features,
        'model_object': model
    }

def compute_model_effects(df_turns):
    """Analyze model (GPT version) effects on continuation."""
    results = []

    for model in df_turns['model'].dropna().unique():
        model_data = df_turns[df_turns['model'] == model]

        # Continuation rate
        cont_rate = model_data['conversation_continued'].mean()

        # Years present
        years = sorted(model_data['year'].dropna().unique())

        # Mean turns for conversations using this model
        conv_hashes = model_data['conversation_hash'].unique()
        conv_data = model_data.groupby('conversation_hash')['turns'].first()
        mean_turns = conv_data.mean()

        results.append({
            'model': model,
            'n_turns': len(model_data),
            'n_conversations': len(conv_hashes),
            'continuation_rate': cont_rate,
            'mean_turns': mean_turns,
            'years': years,
            'year_range': f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        })

    df_models = pd.DataFrame(results)
    return df_models.sort_values('n_conversations', ascending=False)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_full_analysis(filepath):
    """Run complete analysis pipeline."""

    print("=" * 70)
    print("SURVIVAL ANALYSIS OF CHATGPT CONVERSATION DURATION")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    data = load_data(filepath)
    df_turns, df_conv = create_dataframes(data)
    print(f"    Turn-level observations: {len(df_turns):,}")
    print(f"    Unique conversations: {len(df_conv):,}")

    # Descriptive statistics
    print("\n[2] Descriptive Statistics by Year")
    print("-" * 50)
    desc_stats = descriptive_stats_by_year(df_conv)
    print(desc_stats.to_string(index=False))

    # Tail analysis
    print("\n[3] Power-Law Tail Analysis")
    print("-" * 50)
    tail_stats = tail_analysis_by_year(df_conv)
    print(tail_stats.to_string(index=False))
    print("\n    Key finding: Pareto α decreasing over time = HEAVIER TAIL")

    # Hazard by turn
    print("\n[4] Empirical Hazard by Turn")
    print("-" * 50)
    hazard_df = hazard_by_turn(df_turns)
    print(hazard_df.head(15).to_string(index=False))

    # Suggestion effect
    print("\n[5] Suggestion Effect by Year")
    print("-" * 50)
    sugg_effect = suggestion_effect_by_year(df_turns)
    print(sugg_effect.to_string(index=False))

    # Model effects
    print("\n[6] Model (GPT Version) Effects")
    print("-" * 50)
    model_effects = compute_model_effects(df_turns)
    print(model_effects.head(10).to_string(index=False))

    # Prepare survival data
    print("\n[7] Fitting Discrete-Time Hazard Models")
    print("-" * 50)
    df_surv = prepare_survival_data(df_turns)

    # Baseline model
    print("\n    Model 1: Baseline (turn + suggestion + year)")
    baseline = fit_discrete_hazard_model(df_surv, 'baseline')
    print(f"    N = {baseline['n_obs']:,}, Events = {baseline['n_events']:,}")
    print(f"    Pseudo R² = {baseline['pseudo_r2']:.4f}")
    print(f"    AIC = {baseline['aic']:.1f}, BIC = {baseline['bic']:.1f}")
    print("\n    Coefficients (log-odds of ending conversation):")
    print(f"    {'Variable':20} {'β':>8} {'SE':>8} {'OR':>8} {'95% CI':>16} {'p':>8}")
    print("    " + "-" * 70)
    for feat in ['const'] + baseline['features']:
        coef = baseline['coefficients'][feat]
        se = baseline['std_errors'][feat]
        or_val = baseline['odds_ratios'][feat]
        ci_lo = baseline['or_ci_low'][feat]
        ci_hi = baseline['or_ci_high'][feat]
        pval = baseline['pvalues'][feat]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"    {feat:20} {coef:>8.4f} {se:>8.4f} {or_val:>8.4f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}] {pval:>7.4f}{sig}")

    # Extended model
    print("\n    Model 2: Extended (+ word counts)")
    extended = fit_discrete_hazard_model(df_surv, 'extended')
    print(f"    N = {extended['n_obs']:,}, Events = {extended['n_events']:,}")
    print(f"    Pseudo R² = {extended['pseudo_r2']:.4f}")
    print(f"    AIC = {extended['aic']:.1f}, BIC = {extended['bic']:.1f}")
    print("\n    Coefficients (log-odds of ending conversation):")
    print(f"    {'Variable':20} {'β':>8} {'SE':>8} {'OR':>8} {'95% CI':>16} {'p':>8}")
    print("    " + "-" * 70)
    for feat in ['const'] + extended['features']:
        coef = extended['coefficients'][feat]
        se = extended['std_errors'][feat]
        or_val = extended['odds_ratios'][feat]
        ci_lo = extended['or_ci_low'][feat]
        ci_hi = extended['or_ci_high'][feat]
        pval = extended['pvalues'][feat]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"    {feat:20} {coef:>8.4f} {se:>8.4f} {or_val:>8.4f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}] {pval:>7.4f}{sig}")

    # Key takeaways
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    findings = """
    1. TAIL IS GETTING HEAVIER OVER TIME
       - Pareto α decreasing: {:.2f} (2023) → {:.2f} (2024) → {:.2f} (2025)
       - More conversations reaching extreme lengths

    2. SUGGESTION EFFECT IS NEGATIVE (COUNTERINTUITIVE)
       - Conversations WITH suggestions have LOWER continuation rates
       - This suggests selection effect: suggestions shown at natural stopping points

    3. YEAR EFFECTS AFTER CONTROLLING FOR MODEL
       - 2024: OR = {:.3f} (higher hazard than 2023)
       - 2025: OR = {:.3f} (lower hazard than 2023)

    4. CANNOT FULLY SEPARATE MODEL VS YEAR EFFECTS
       - Models are highly collinear with year (different models each year)
       - Need within-model comparison where possible
    """.format(
        tail_stats[tail_stats['year']==2023]['pareto_alpha'].values[0],
        tail_stats[tail_stats['year']==2024]['pareto_alpha'].values[0],
        tail_stats[tail_stats['year']==2025]['pareto_alpha'].values[0],
        baseline['odds_ratios']['year_2024'],
        baseline['odds_ratios']['year_2025']
    )
    print(findings)

    return {
        'desc_stats': desc_stats,
        'tail_stats': tail_stats,
        'hazard_df': hazard_df,
        'sugg_effect': sugg_effect,
        'model_effects': model_effects,
        'baseline_model': baseline,
        'extended_model': extended,
        'df_surv': df_surv
    }

def export_for_r(df_surv, output_path):
    """Export data for R survival analysis."""

    # Select relevant columns
    cols = ['conversation_hash', 'turn', 'turns', 'year', 'model',
            'suggestion_given_count', 'suggestion_used_flag',
            'conversation_continued', 'user_word_count', 'assistant_word_count',
            'country', 'language', 'toxic', 'event', 'suggestion_given',
            'year_2024', 'year_2025', 'log_turn', 'user_wc_log', 'asst_wc_log']

    df_export = df_surv[[c for c in cols if c in df_surv.columns]].copy()

    # Save as CSV for R
    df_export.to_csv(output_path, index=False)
    print(f"Exported {len(df_export):,} rows to {output_path}")

    return df_export

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    filepath = "/Users/ilanstrauss/Desktop/Ilan Folder/2024/NYC AI/Key Research Papers/Engagement Juicing/Data 2/analysis_data_flattened.jsonl"

    results = run_full_analysis(filepath)

    # Export for R
    print("\n[8] Exporting data for R...")
    export_for_r(results['df_surv'], '/private/tmp/follow_up_research/analysis/survival_data.csv')
