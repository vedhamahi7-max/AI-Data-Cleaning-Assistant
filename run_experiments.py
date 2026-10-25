"""
Experiment Runner for Capstone 65: AI Data Cleaning Assistant
Runs ChatGPT Zero-Shot, Gemini Engineered, Reusable Pipeline, and Human Ground Truth
against the messy public benchmark dataset, capturing detailed comparative metrics and bug logs.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from ai_code_replicas import clean_with_chatgpt_zero_shot, clean_with_gemini_engineered
from cleaner import AIDataCleaningAssistant

RAW_PATH = Path("data/raw/messy_customer_data.csv")
GT_PATH = Path("data/cleaned/manual_cleaned_data.csv")
OUT_ZERO_PATH = Path("data/cleaned/ai_zero_shot_cleaned.csv")
OUT_ENG_PATH = Path("data/cleaned/ai_engineered_cleaned.csv")
OUT_PIPE_PATH = Path("data/cleaned/pipeline_cleaned_data.csv")
METRICS_PATH = Path("reports/comparison_metrics.json")
BUG_LOG_PATH = Path("reports/bug_log.csv")


def run_all_experiments():
    print("=" * 70)
    print("RUNNING CAPSTONE 65 EXPERIMENTAL SUITE: AI RELIABILITY BENCHMARK")
    print("=" * 70)

    # 1. Load Datasets
    df_raw = pd.read_csv(RAW_PATH)
    df_gt = pd.read_csv(GT_PATH)
    print(f"Loaded Raw Dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    print(f"Loaded Ground Truth: {len(df_gt)} rows")

    # 2. Run ChatGPT Zero-Shot
    print("\n--- 1. Running ChatGPT Zero-Shot Pipeline ---")
    df_zero, zero_notes = clean_with_chatgpt_zero_shot(df_raw)
    df_zero.to_csv(OUT_ZERO_PATH, index=False)
    print(f"ChatGPT Zero-Shot finished. Retained: {len(df_zero)} rows ({len(df_zero)/len(df_raw)*100:.1f}%)")

    # 3. Run Gemini Prompt-Engineered
    print("\n--- 2. Running Gemini Prompt-Engineered Pipeline ---")
    df_eng, eng_notes = clean_with_gemini_engineered(df_raw)
    df_eng.to_csv(OUT_ENG_PATH, index=False)
    print(f"Gemini Engineered finished. Retained: {len(df_eng)} rows ({len(df_eng)/len(df_raw)*100:.1f}%)")

    # 4. Run Reusable Production Pipeline
    print("\n--- 3. Running Automated Reusable Pipeline ---")
    cleaner = AIDataCleaningAssistant()
    df_pipe, pipe_audit = cleaner.clean(df_raw)
    df_pipe.to_csv(OUT_PIPE_PATH, index=False)
    print(f"Reusable Pipeline finished. Retained: {len(df_pipe)} rows ({len(df_pipe)/len(df_raw)*100:.1f}%)")

    # 5. Compute Deep Evaluation Metrics
    print("\n--- 4. Computing Reliability & Data Integrity Metrics ---")
    candidates = {
        "ChatGPT Zero-Shot": df_zero,
        "Gemini Prompt-Engineered": df_eng,
        "Reusable Pipeline": df_pipe,
        "Manual Ground Truth": df_gt,
    }

    metrics_summary = {}

    for name, df_cand in candidates.items():
        # Metric A: Row Retention Rate
        retention_pct = round(len(df_cand) / len(df_raw) * 100, 2)

        # Metric B: Negative Refund Preservation
        # Ground truth has negative values representing refunds
        refund_count = int((df_cand["purchase_amount"] < 0).sum())
        gt_refund_count = int((df_gt["purchase_amount"] < 0).sum())
        refund_preserved_pct = round(refund_count / max(1, gt_refund_count) * 100, 2) if refund_count <= gt_refund_count else 100.0

        # Metric C: Name Corruption Rate (Checking for preserved apostrophes and hyphens)
        sample_names = df_cand["full_name"].dropna().astype(str)
        has_hyphen_or_apos = sample_names.str.contains(r"[-']").sum()
        gt_hyphen_or_apos = df_gt["full_name"].dropna().astype(str).str.contains(r"[-']").sum()
        special_char_name_integrity = round(has_hyphen_or_apos / max(1, gt_hyphen_or_apos) * 100, 2)

        # Metric D: Date Swap / Parsing Integrity
        # Valid ISO format percentage
        dates = df_cand["signup_date"].dropna().astype(str)
        valid_iso_dates = dates.str.match(r"^\d{4}-\d{2}-\d{2}$").sum()
        date_iso_pct = round(valid_iso_dates / max(1, len(dates)) * 100, 2)

        # Metric E: Statistical Distribution Drift vs Ground Truth (Annual Income & Age)
        # Using Kolmogorov-Smirnov 2-sample test and Wasserstein Distance
        inc_cand = df_cand["annual_income"].dropna().values
        inc_gt = df_gt["annual_income"].dropna().values
        ks_stat_inc, ks_pval_inc = ks_2samp(inc_cand, inc_gt)
        wd_inc = float(wasserstein_distance(inc_cand, inc_gt))

        age_cand = df_cand["age"].dropna().values
        age_gt = df_gt["age"].dropna().values
        ks_stat_age, ks_pval_age = ks_2samp(age_cand, age_gt)
        wd_age = float(wasserstein_distance(age_cand, age_gt))

        # Metric F: Categorical Validity (Gender & Country recognized canonical standards)
        canonical_genders = {"Female", "Male", "Non-Binary", "Prefer not to say"}
        valid_gender_pct = round(df_cand["gender"].isin(canonical_genders).sum() / max(1, len(df_cand)) * 100, 2)

        canonical_countries = {"United States", "United Kingdom", "Canada", "Germany", "Australia", "Other"}
        valid_country_pct = round(df_cand["country"].isin(canonical_countries).sum() / max(1, len(df_cand)) * 100, 2)

        # Metric G: Overall Composite Reliability Score (0 to 100)
        # Penalties: severe dropna losses, name corruption, refund erasure, distribution drift
        score = 100.0
        if retention_pct < 85:
            score -= (85 - retention_pct) * 1.5
        if refund_preserved_pct < 50:
            score -= 20.0
        if special_char_name_integrity < 50:
            score -= 15.0
        if valid_gender_pct < 90:
            score -= 10.0
        if ks_pval_inc < 0.01:
            score -= 10.0  # Significant distribution distortion
        if wd_age > 5.0:
            score -= 10.0

        composite_reliability = max(10.0, min(100.0, round(score, 1)))

        metrics_summary[name] = {
            "total_rows": len(df_cand),
            "row_retention_pct": retention_pct,
            "refund_transactions_retained": refund_count,
            "refund_retention_pct": refund_preserved_pct,
            "name_special_char_integrity_pct": special_char_name_integrity,
            "date_iso_conformance_pct": date_iso_pct,
            "gender_canonical_pct": valid_gender_pct,
            "country_canonical_pct": valid_country_pct,
            "income_wasserstein_distance": round(wd_inc, 2),
            "income_ks_pvalue": round(float(ks_pval_inc), 4),
            "age_wasserstein_distance": round(wd_age, 2),
            "age_ks_pvalue": round(float(ks_pval_age), 4),
            "composite_reliability_score": composite_reliability,
        }

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved comparison metrics to: {METRICS_PATH}")

    # 6. Generate Bug Log CSV
    bugs = [
        {
            "bug_id": "BUG-001",
            "model_affected": "ChatGPT (Zero-Shot)",
            "category": "Catastrophic Silent Data Loss",
            "code_snippet": "df.dropna(inplace=True)",
            "impact": "Drops ~35-45% of entire dataset because optional fields ('phone_number') contain nulls.",
            "severity": "Critical",
            "remedy": "Specify crucial key subsets (subset=['customer_id']) and retain rows with optional nulls."
        },
        {
            "bug_id": "BUG-002",
            "model_affected": "ChatGPT (Zero-Shot)",
            "category": "Date Format Transposition (Day/Month Swap)",
            "code_snippet": "pd.to_datetime(df['signup_date'])",
            "impact": "Assumes American MM/DD/YYYY format. Inverts international dates (e.g. 05/04/2023 becomes May 4th instead of April 5th) and generates NaT on days > 12.",
            "severity": "High",
            "remedy": "Pass dayfirst=True or use multi-format cascading regex parsers."
        },
        {
            "bug_id": "BUG-003",
            "model_affected": "ChatGPT (Zero-Shot)",
            "category": "Data Corruption via Overzealous Regex",
            "code_snippet": "str.replace(r'[^a-zA-Z\\s]', '', regex=True)",
            "impact": "Mutilates valid patronymic and compound names (e.g., 'O\\'Connor' -> 'OConnor', 'Jean-Luc' -> 'JeanLuc').",
            "severity": "Medium",
            "remedy": "Allow internal hyphens and apostrophes: r'[^a-zA-Z\\s\\-\\']'."
        },
        {
            "bug_id": "BUG-004",
            "model_affected": "ChatGPT (Zero-Shot)",
            "category": "Domain Logic Failure (Financial Refund Annihilation)",
            "code_snippet": "df['purchase_amount'] = df['purchase_amount'].abs()",
            "impact": "Treats legitimate negative refund/return transactions as errors, converting -$85 into +$85 and inflating gross revenue.",
            "severity": "Critical",
            "remedy": "Preserve negative signs while stripping currency symbols, distinguishing returns from errors."
        },
        {
            "bug_id": "BUG-005",
            "model_affected": "ChatGPT (Zero-Shot) & Gemini (Engineered)",
            "category": "Statistical Distortion (Responsible AI Imputation Bias)",
            "code_snippet": "df['annual_income'].fillna(df['annual_income'].mean())",
            "impact": "Global mean imputation collapses variance, artificially peaks the distribution, and skews underrepresented demographics.",
            "severity": "High",
            "remedy": "Use subgroup-aware imputation (by Country & Age Group) with an imputation indicator column."
        },
        {
            "bug_id": "BUG-006",
            "model_affected": "ChatGPT (Zero-Shot)",
            "category": "Parsing Syntax Failure (Multiplier Ignorance)",
            "code_snippet": "pd.to_numeric(df['annual_income'].str.replace('$',''))",
            "impact": "Fails to parse shorthand values like '75k' or '120k', silently turning valid salaries into NaNs.",
            "severity": "Medium",
            "remedy": "Detect 'k'/'m' suffixes and multiply by 1,000 or 1,000,000."
        },
        {
            "bug_id": "BUG-007",
            "model_affected": "Gemini (Engineered)",
            "category": "Categorical Misclassification & Unmapped Values",
            "code_snippet": "df['country'].map(country_lookup).fillna('Other')",
            "impact": "Uncommon international aliases ('Deutschland', 'Oz') are cast to 'Other' if dictionary lacks exhaustive coverage.",
            "severity": "Low",
            "remedy": "Include fuzzy matching or ISO 3166-1 alpha-2 standard country codes."
        },
        {
            "bug_id": "BUG-008",
            "model_affected": "Both AI Models",
            "category": "Unchecked Deduplication on Dirty Keys",
            "code_snippet": "df.drop_duplicates(subset=['customer_id'])",
            "impact": "Fails to identify duplicates with leading/trailing whitespaces (e.g. ' CUST-1042 ' vs 'CUST-1042').",
            "severity": "Medium",
            "remedy": "Perform key normalization (strip and uppercase) BEFORE executing deduplication."
        }
    ]

    df_bugs = pd.DataFrame(bugs)
    df_bugs.to_csv(BUG_LOG_PATH, index=False)
    print(f"Saved structured bug log to: {BUG_LOG_PATH} ({len(df_bugs)} documented bugs)")

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS TABLE:")
    print("=" * 70)
    metrics_df = pd.DataFrame.from_dict(metrics_summary, orient="index")
    print(metrics_df[["total_rows", "row_retention_pct", "refund_retention_pct", "income_ks_pvalue", "composite_reliability_score"]])
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
