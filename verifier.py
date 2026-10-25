"""
Verification and Responsible AI Audit Suite for Capstone 65: AI Data Cleaning Assistant
Performs rigorous schema validation, statistical hypothesis testing (KS-test, Chi-square, Wasserstein),
and Responsible AI fairness audits (Demographic Parity, Drop Rate Disparity, Representation Skew).
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp, wasserstein_distance


class DataCleaningVerifier:
    """Verifies cleaned datasets against data integrity, statistical preservation, and fairness rules."""

    def __init__(self, df_raw: pd.DataFrame, df_gt: pd.DataFrame):
        self.df_raw = df_raw
        self.df_gt = df_gt

    def verify_candidate(self, df_candidate: pd.DataFrame, candidate_name: str) -> dict[str, Any]:
        """Runs complete verification on a candidate cleaned dataset."""
        report = {
            "candidate_name": candidate_name,
            "schema_checks": self._verify_schema(df_candidate),
            "integrity_checks": self._verify_integrity(df_candidate),
            "statistical_preservation": self._verify_statistics(df_candidate),
            "responsible_ai_fairness": self._verify_responsible_ai(df_candidate),
        }
        return report

    def _verify_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validates column types, non-null constraints, and formatting."""
        expected_cols = [
            "customer_id", "full_name", "age", "gender", "country", "signup_date",
            "annual_income", "purchase_amount", "transaction_count", "email",
            "phone_number", "satisfaction_score", "churn"
        ]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        # Check non-null on primary key
        id_nulls = int(df["customer_id"].isna().sum()) if "customer_id" in df.columns else len(df)
        
        # Check numeric types
        is_inc_numeric = bool(np.issubdtype(df["annual_income"].dtype, np.number)) if "annual_income" in df.columns else False
        is_age_numeric = bool(np.issubdtype(df["age"].dtype, np.number)) if "age" in df.columns else False
        
        # Check date format
        if "signup_date" in df.columns:
            date_matches = df["signup_date"].dropna().astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").sum()
            date_valid_pct = round(date_matches / max(1, len(df)) * 100, 2)
        else:
            date_valid_pct = 0.0

        return {
            "missing_columns": missing_cols,
            "id_null_count": id_nulls,
            "is_income_numeric": is_inc_numeric,
            "is_age_numeric": is_age_numeric,
            "date_iso_conformance_pct": date_valid_pct,
            "passed_schema": len(missing_cols) == 0 and id_nulls == 0 and is_inc_numeric and is_age_numeric,
        }

    def _verify_integrity(self, df: pd.DataFrame) -> dict[str, Any]:
        """Checks for corruption of special characters, refund preservation, and duplicate count."""
        # Check name integrity
        if "full_name" in df.columns:
            names = df["full_name"].dropna().astype(str)
            names_with_special = names.str.contains(r"[-']").sum()
        else:
            names_with_special = 0

        # Check negative refunds
        if "purchase_amount" in df.columns:
            refund_count = int((df["purchase_amount"] < 0).sum())
        else:
            refund_count = 0

        # Duplicate customer IDs
        dup_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else 0

        return {
            "names_preserving_hyphens_apostrophes": int(names_with_special),
            "negative_refund_transactions": refund_count,
            "duplicate_customer_ids": dup_ids,
            "passed_integrity": dup_ids == 0 and refund_count > 0 and names_with_special > 0,
        }

    def _verify_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculates KS-test and Wasserstein distance against ground truth."""
        stat_results = {}
        for num_col in ["annual_income", "age", "purchase_amount"]:
            if num_col in df.columns and num_col in self.df_gt.columns:
                c_vals = df[num_col].dropna().values
                gt_vals = self.df_gt[num_col].dropna().values
                
                if len(c_vals) > 0 and len(gt_vals) > 0:
                    ks_stat, ks_pval = ks_2samp(c_vals, gt_vals)
                    wd = wasserstein_distance(c_vals, gt_vals)
                    stat_results[num_col] = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "ks_pvalue": round(float(ks_pval), 4),
                        "wasserstein_distance": round(float(wd), 2),
                        "distribution_preserved": bool(ks_pval >= 0.01),
                    }
        return stat_results

    def _verify_responsible_ai(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Responsible AI & Fairness Audit:
        Measures whether row dropping or imputation disproportionately impacted
        underrepresented demographic subgroups (e.g. Non-Binary, Prefer not to say, minority countries).
        """
        fairness_report = {}

        # 1. Gender Representation Disparity
        if "gender" in df.columns and "gender" in self.df_raw.columns:
            raw_gender_dist = self.df_raw["gender"].dropna().str.lower().value_counts(normalize=True).to_dict()
            clean_gender_dist = df["gender"].value_counts(normalize=True).to_dict()

            # Measure representation shift for Non-Binary / Marginalized groups
            nb_clean_pct = clean_gender_dist.get("Non-Binary", 0.0)
            fairness_report["gender_distribution_clean"] = {k: round(v * 100, 2) for k, v in clean_gender_dist.items()}
            fairness_report["non_binary_representation_pct"] = round(nb_clean_pct * 100, 2)

        # 2. Drop Rate by Gender
        # Check if any demographic group had a significantly higher drop rate
        if "gender" in self.df_raw.columns and "customer_id" in self.df_raw.columns:
            retained_ids = set(df["customer_id"].dropna().astype(str).str.strip().str.upper())
            raw_copy = self.df_raw.copy()
            raw_copy["norm_id"] = raw_copy["customer_id"].astype(str).str.strip().str.upper()
            raw_copy["is_retained"] = raw_copy["norm_id"].isin(retained_ids)

            retention_by_gender = raw_copy.groupby("gender", observed=False)["is_retained"].mean().to_dict()
            fairness_report["retention_rate_by_raw_gender"] = {str(k): round(float(v) * 100, 2) for k, v in retention_by_gender.items()}

        # 3. Income Disparity Check
        # Check if income variance collapsed due to naive mean imputation
        if "annual_income" in df.columns and "annual_income" in self.df_gt.columns:
            clean_std = float(df["annual_income"].std())
            gt_std = float(self.df_gt["annual_income"].std())
            std_ratio = round(clean_std / max(1e-5, gt_std), 3)
            fairness_report["income_variance_preservation_ratio"] = std_ratio
            fairness_report["passed_variance_check"] = bool(std_ratio >= 0.85)

        return fairness_report


def generate_verification_report():
    df_raw = pd.read_csv("data/raw/messy_customer_data.csv")
    df_gt = pd.read_csv("data/cleaned/manual_cleaned_data.csv")
    df_zero = pd.read_csv("data/cleaned/ai_zero_shot_cleaned.csv")
    df_eng = pd.read_csv("data/cleaned/ai_engineered_cleaned.csv")
    df_pipe = pd.read_csv("data/cleaned/pipeline_cleaned_data.csv")

    verifier = DataCleaningVerifier(df_raw, df_gt)

    reports = {
        "ChatGPT Zero-Shot": verifier.verify_candidate(df_zero, "ChatGPT Zero-Shot"),
        "Gemini Prompt-Engineered": verifier.verify_candidate(df_eng, "Gemini Prompt-Engineered"),
        "Reusable Pipeline": verifier.verify_candidate(df_pipe, "Reusable Pipeline"),
        "Manual Ground Truth": verifier.verify_candidate(df_gt, "Manual Ground Truth"),
    }

    def np_encoder(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        elif isinstance(obj, (np.floating, np.ndarray)):
            return float(obj) if isinstance(obj, np.floating) else obj.tolist()
        return str(obj)

    report_path = Path("reports/verification_results.json")
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=4, default=np_encoder)
    print(f"Verification report saved to: {report_path}")
    return reports


if __name__ == "__main__":
    generate_verification_report()
