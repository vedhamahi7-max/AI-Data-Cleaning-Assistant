"""
Dataset Generator for Capstone 65: AI Data Cleaning Assistant
Generates a realistic, messy public e-commerce customer transaction dataset
with documented dirty data anomalies, edge cases, and known ground truth.
"""

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

# Set fixed seeds for deterministic reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_RAW_PATH = Path("data/raw/messy_customer_data.csv")
OUTPUT_DICT_PATH = Path("data/raw/data_dictionary.json")
OUTPUT_MANUAL_PATH = Path("data/cleaned/manual_cleaned_data.csv")


def generate_messy_dataset(n_samples: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a raw messy DataFrame and its corresponding manual ground truth DataFrame."""
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
        "Jean-Luc", "O'Connor", "Renée", "Chloë", "D'Angelo"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell"
    ]

    raw_records = []
    ground_truth_records = []

    for i in range(1, n_samples + 1):
        # 1. Base clean attributes
        cust_id_clean = f"CUST-{1000 + i}"
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name_clean = f"{fn} {ln}"
        
        # Age distribution: 18 to 75
        age_clean = int(np.random.normal(loc=39, scale=12))
        age_clean = max(18, min(80, age_clean))
        
        # Gender
        gender_weights = [0.47, 0.47, 0.04, 0.02]
        gender_choices = ["Female", "Male", "Non-Binary", "Prefer not to say"]
        gender_clean = random.choices(gender_choices, weights=gender_weights)[0]
        
        # Country
        country_choices = ["United States", "United Kingdom", "Canada", "Germany", "Australia"]
        country_clean = random.choice(country_choices)
        
        # Signup Date (Between 2021-01-01 and 2024-06-30)
        days_offset = random.randint(0, 1270)
        signup_date_clean = (pd.Timestamp("2021-01-01") + pd.Timedelta(days=days_offset)).strftime("%Y-%m-%d")
        
        # Annual Income
        income_clean = round(float(np.random.normal(loc=65000, scale=22000)), 2)
        income_clean = max(18000.0, income_clean)
        
        # Purchase Amount (90% purchases, 10% legitimate returns/refunds)
        is_return = random.random() < 0.08
        if is_return:
            purchase_clean = -round(random.uniform(15.0, 350.0), 2)
        else:
            purchase_clean = round(float(np.random.exponential(scale=140.0) + 12.0), 2)
            
        # Transaction Count
        tx_count_clean = max(1, int(np.random.poisson(lam=6)))
        
        clean_fn = fn.lower().replace("-", "").replace("'", "")
        clean_ln = ln.lower().replace("-", "").replace("'", "")
        email_prefix = f"{clean_fn}.{clean_ln}"
        domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]
        email_clean = f"{email_prefix}@{random.choice(domains)}"
        
        # Phone
        phone_clean = f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        
        # Satisfaction Score (1 to 5)
        satisfaction_clean = random.choices([1, 2, 3, 4, 5], weights=[0.08, 0.12, 0.25, 0.35, 0.20])[0]
        
        # Churn Status
        churn_clean = random.choices(["No", "Yes"], weights=[0.74, 0.26])[0]

        # -------------------------------------------------------------
        # Inject Dirty Noise into Raw Record
        # -------------------------------------------------------------
        # Customer ID
        p_id = random.random()
        if p_id < 0.03:
            cust_id_raw = f"  {cust_id_clean}  "  # Whitespace
        elif p_id < 0.05:
            cust_id_raw = cust_id_clean.lower()  # Lowercase
        elif p_id < 0.07:
            cust_id_raw = np.nan  # Missing ID
        else:
            cust_id_raw = cust_id_clean

        # Name
        p_name = random.random()
        if p_name < 0.06:
            name_raw = name_clean.upper()
        elif p_name < 0.12:
            name_raw = name_clean.lower()
        elif p_name < 0.16:
            name_raw = f"  {name_clean}   "
        elif p_name < 0.19:
            title = random.choice(["Dr. ", "Mr. ", "Ms. ", "Prof. "])
            name_raw = title + name_clean
        elif p_name < 0.22:
            name_raw = np.nan
        else:
            name_raw = name_clean

        # Age
        p_age = random.random()
        if p_age < 0.04:
            age_raw = str(age_clean)  # Stored as string
        elif p_age < 0.06:
            age_raw = f" {age_clean} yrs"
        elif p_age < 0.08:
            age_raw = -abs(age_clean)  # Erroneous negative age
        elif p_age < 0.10:
            age_raw = random.choice([195, 999, -5])  # Extreme outlier
        elif p_age < 0.16:
            age_raw = np.nan  # Missing
        else:
            age_raw = age_clean

        # Gender
        p_gender = random.random()
        if p_gender < 0.15:
            if gender_clean == "Female":
                gender_raw = random.choice(["F", "f", "female", "FEMALE", " Fem "])
            elif gender_clean == "Male":
                gender_raw = random.choice(["M", "m", "male", "MALE", " Mal "])
            elif gender_clean == "Non-Binary":
                gender_raw = random.choice(["NB", "non-binary", "Non Binary"])
            else:
                gender_raw = "N/A"
        elif p_gender < 0.20:
            gender_raw = np.nan
        else:
            gender_raw = gender_clean

        # Country
        p_country = random.random()
        if p_country < 0.18:
            country_mapping = {
                "United States": random.choice(["US", "USA", "U.S.A.", "united states", "usa", "U.S."]),
                "United Kingdom": random.choice(["UK", "uk", "U.K.", "Great Britain", "GB"]),
                "Canada": random.choice(["CA", "ca", "CAN", "canada"]),
                "Germany": random.choice(["DE", "Deutschland", "germany", "Ger"]),
                "Australia": random.choice(["AU", "AUS", "australia", "Oz"])
            }
            country_raw = country_mapping.get(country_clean, country_clean)
        elif p_country < 0.22:
            country_raw = np.nan
        else:
            country_raw = country_clean

        # Signup Date
        p_date = random.random()
        dt_obj = pd.Timestamp(signup_date_clean)
        if p_date < 0.10:
            # European format DD/MM/YYYY (AI trap: pd.to_datetime may parse as MM/DD/YYYY)
            date_raw = dt_obj.strftime("%d/%m/%Y")
        elif p_date < 0.20:
            date_raw = dt_obj.strftime("%m-%d-%Y")
        elif p_date < 0.25:
            date_raw = dt_obj.strftime("%Y.%m.%d")
        elif p_date < 0.30:
            date_raw = dt_obj.strftime("%d %b %Y")
        elif p_date < 0.33:
            date_raw = "2023-02-30"  # Invalid calendar date
        elif p_date < 0.36:
            date_raw = random.choice(["yesterday", "N/A", "unknown", "NULL"])
        elif p_date < 0.40:
            date_raw = np.nan
        else:
            date_raw = signup_date_clean

        # Annual Income
        p_inc = random.random()
        if p_inc < 0.08:
            income_raw = f"${income_clean:,.2f}"  # Currency symbol + comma
        elif p_inc < 0.14:
            income_raw = f"USD {income_clean:,.0f}"
        elif p_inc < 0.18:
            income_raw = f"{income_clean / 1000:.0f}k"  # Shorthand 75k
        elif p_inc < 0.21:
            income_raw = -abs(income_clean)  # Corrupted negative income
        elif p_inc < 0.24:
            income_raw = random.choice(["Confidential", "none", "unknown", "0.00"])
        elif p_inc < 0.30:
            income_raw = np.nan
        else:
            income_raw = income_clean

        # Purchase Amount
        p_pur = random.random()
        if p_pur < 0.08:
            purchase_raw = f"${purchase_clean:,.2f}" if purchase_clean >= 0 else f"-${abs(purchase_clean):,.2f}"
        elif p_pur < 0.12:
            purchase_raw = f"EUR {purchase_clean:.2f}"
        elif p_pur < 0.15:
            purchase_raw = 9999999.00  # Extreme typo outlier
        elif p_pur < 0.18:
            purchase_raw = random.choice(["Free", "refunded", "cancelled", "N/A"])
        elif p_pur < 0.24:
            purchase_raw = np.nan
        else:
            purchase_raw = purchase_clean

        # Transaction Count
        p_tx = random.random()
        if p_tx < 0.05:
            tx_count_raw = str(tx_count_clean)
        elif p_tx < 0.08:
            tx_count_raw = -tx_count_clean  # Negative transactions
        elif p_tx < 0.10:
            tx_count_raw = 999  # Outlier
        elif p_tx < 0.15:
            tx_count_raw = np.nan
        else:
            tx_count_raw = tx_count_clean

        # Email
        p_em = random.random()
        if p_em < 0.08:
            email_raw = email_clean.upper()
        elif p_em < 0.12:
            email_raw = f"  {email_clean}  "
        elif p_em < 0.16:
            email_raw = email_clean.replace("@", "")  # Invalid missing @
        elif p_em < 0.20:
            email_raw = random.choice(["no_email", "none@none", "test", "N/A"])
        elif p_em < 0.25:
            email_raw = np.nan
        else:
            email_raw = email_clean

        # Phone Number (High missingness: 38% missing - intentionally tests naive dropna)
        p_ph = random.random()
        if p_ph < 0.38:
            phone_raw = np.nan
        elif p_ph < 0.48:
            phone_raw = phone_clean.replace("-", "")
        elif p_ph < 0.55:
            phone_raw = f"({phone_clean[3:6]}) {phone_clean[7:10]}-{phone_clean[11:]}"
        else:
            phone_raw = phone_clean

        # Satisfaction Score
        p_sat = random.random()
        if p_sat < 0.06:
            sat_raw = f"{satisfaction_clean}/5"
        elif p_sat < 0.10:
            sat_score_words = {1: "Very Poor", 2: "Poor", 3: "Average", 4: "Good", 5: "Excellent"}
            sat_raw = sat_score_words[satisfaction_clean]
        elif p_sat < 0.14:
            sat_raw = random.choice([0, 10, -1])  # Out of range
        elif p_sat < 0.20:
            sat_raw = np.nan
        else:
            sat_raw = satisfaction_clean

        # Churn Status
        p_ch = random.random()
        if p_ch < 0.08:
            churn_raw = 1 if churn_clean == "Yes" else 0
        elif p_ch < 0.14:
            churn_raw = "TRUE" if churn_clean == "Yes" else "FALSE"
        elif p_ch < 0.18:
            churn_raw = churn_clean.lower()
        elif p_ch < 0.23:
            churn_raw = np.nan
        else:
            churn_raw = churn_clean

        raw_record = {
            "customer_id": cust_id_raw,
            "full_name": name_raw,
            "age": age_raw,
            "gender": gender_raw,
            "country": country_raw,
            "signup_date": date_raw,
            "annual_income": income_raw,
            "purchase_amount": purchase_raw,
            "transaction_count": tx_count_raw,
            "email": email_raw,
            "phone_number": phone_raw,
            "satisfaction_score": sat_raw,
            "churn": churn_raw,
        }
        raw_records.append(raw_record)

        # Ground Truth Cleaned Record
        gt_record = {
            "customer_id": cust_id_clean,
            "full_name": name_clean,
            "age": age_clean,
            "gender": gender_clean,
            "country": country_clean,
            "signup_date": signup_date_clean,
            "annual_income": income_clean,
            "purchase_amount": purchase_clean,
            "transaction_count": tx_count_clean,
            "email": email_clean,
            "phone_number": phone_clean,
            "satisfaction_score": satisfaction_clean,
            "churn": churn_clean,
        }
        ground_truth_records.append(gt_record)

    df_raw = pd.DataFrame(raw_records)
    df_gt = pd.DataFrame(ground_truth_records)

    # Inject 45 duplicate rows into the raw dataset to test duplicate deduplication logic
    dup_indices = random.sample(range(len(df_raw)), 45)
    duplicates = df_raw.iloc[dup_indices].copy()
    # Modify casing on a few duplicates to test case-insensitive deduplication
    duplicates.loc[duplicates.index[:15], "full_name"] = duplicates.loc[duplicates.index[:15], "full_name"].astype(str).str.lower()
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

    # Inject 15 empty rows (entirely NaN)
    empty_rows = pd.DataFrame([{col: np.nan for col in df_raw.columns} for _ in range(15)])
    df_raw = pd.concat([df_raw, empty_rows], ignore_index=True)

    # Shuffle raw dataframe
    df_raw = df_raw.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return df_raw, df_gt


def save_datasets():
    OUTPUT_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_raw, df_gt = generate_messy_dataset(n_samples=1200)

    df_raw.to_csv(OUTPUT_RAW_PATH, index=False)
    df_gt.to_csv(OUTPUT_MANUAL_PATH, index=False)

    data_dictionary = {
        "dataset_name": "International E-Commerce Customer Analytics & Demographics (Messy Benchmark)",
        "total_raw_rows": len(df_raw),
        "total_ground_truth_rows": len(df_gt),
        "fields": {
            "customer_id": "Unique customer identifier string (CUST-XXXX). Contains whitespace, lowercase, missing IDs.",
            "full_name": "Customer first and last name. Contains titles, apostrophes (O'Connor), hyphens (Jean-Luc), casing anomalies.",
            "age": "Customer age in years. Contains strings ('45 yrs'), negative numbers (-39), extreme outliers (195, 999), NaNs.",
            "gender": "Self-reported gender. Inconsistent categories ('Female', 'F', 'f', 'Male', 'M', 'Non-Binary', 'NB').",
            "country": "Customer country of residence. Inconsistent naming ('USA', 'U.S.A.', 'US', 'Great Britain', 'UK').",
            "signup_date": "Account creation timestamp. Mixed date formats (ISO, DD/MM/YYYY, MM-DD-YYYY, text 'yesterday', invalid '2023-02-30').",
            "annual_income": "Estimated annual salary in USD. Dirty currencies ('$52,000', 'USD 75000', '75k', negative values, strings).",
            "purchase_amount": "Recent transaction total in USD. Valid negative values (refunds/returns), extreme outliers, text ('Free', 'N/A').",
            "transaction_count": "Lifetime transaction count. Numeric and string artifacts, negative counts, extreme outliers.",
            "email": "Customer contact email. Missing '@' symbols, trailing spaces, uppercase, dummy placeholders.",
            "phone_number": "Contact phone. High missingness (38% MCAR), multiple punctuation styles.",
            "satisfaction_score": "Customer survey rating (1 to 5). Contains text ('Excellent', '5/5'), out-of-bounds numbers (-1, 10).",
            "churn": "Binary churn label ('Yes' / 'No'). Stored inconsistently as 1/0, TRUE/FALSE, lowercase strings."
        },
        "injected_bugs_to_test": [
            "Naive dropna() deleting valid rows due to phone_number missingness",
            "pd.to_datetime European day-first swap flipping month and day",
            "Regex stripping hyphens/apostrophes in names",
            "Treating negative purchase amounts as errors rather than legitimate refunds",
            "Distortion of sub-population distributions via blind mean imputation",
            "Type-coercion errors when casting '$1,200.50' directly with astype(float)",
            "Deduplication failing due to leading/trailing whitespace"
        ]
    }

    with open(OUTPUT_DICT_PATH, "w") as f:
        json.dump(data_dictionary, f, indent=4)

    print(f"Generated raw dataset: {OUTPUT_RAW_PATH} ({len(df_raw)} rows, {len(df_raw.columns)} columns)")
    print(f"Generated ground truth: {OUTPUT_MANUAL_PATH} ({len(df_gt)} rows)")
    print(f"Saved data dictionary: {OUTPUT_DICT_PATH}")


if __name__ == "__main__":
    save_datasets()
