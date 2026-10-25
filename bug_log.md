# Capstone 65: AI Data Cleaning Assistant — Bug Log & Failure Taxonomy

## Executive Summary
This document provides an exhaustive post-mortem and empirical taxonomy of bugs, hallucinations, logic errors, and silent failure modes observed in AI-generated Python data-cleaning code. The evaluation was conducted across two distinct generative AI workflows:
1. **ChatGPT (Free Tier / Zero-Shot)**: Prompted with standard colloquial data-cleaning requests (*"Write Python pandas code to clean this dataset"*).
2. **Gemini Flash (Prompt-Engineered)**: Prompted with few-shot schema constraints, explicit domain guardrails, and Chain-of-Thought (CoT) instructions.

---

## Summary Matrix of Discovered Bugs

| Bug ID | Failure Category | Model Affected | Severity | Root Cause | Business / Data Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BUG-001** | Catastrophic Silent Data Loss | ChatGPT (Zero-Shot) | **Critical (P0)** | Blind invocation of `df.dropna(inplace=True)` | **98.7% of valid dataset erased** due to non-critical missingness in optional contact fields (`phone_number`). |
| **BUG-002** | Chronological Inversion | ChatGPT (Zero-Shot) | **High (P1)** | Naive `pd.to_datetime` without `dayfirst=True` | Swaps European day and month (`05/04/2023` becomes May 4th instead of April 5th); generates `NaT` on days > 12. |
| **BUG-003** | Data Corruption / Overzealous Regex | ChatGPT (Zero-Shot) | **Medium (P2)** | Regex `r'[^a-zA-Z\s]'` stripping non-alphabetics | Mutilates valid patronymic and hyphenated names (`O'Connor` &rarr; `OConnor`, `Jean-Luc` &rarr; `JeanLuc`). |
| **BUG-004** | Financial Logic Inversion | ChatGPT (Zero-Shot) | **Critical (P0)** | Outlier filtering via `df['purchase_amount'].abs()` | Legitimate negative transactions (merchandise refunds / returns) converted to positive sales, inflating revenue. |
| **BUG-005** | Statistical Distortion & Demographic Bias | ChatGPT & Gemini | **High (P1)** | Global mean/median imputation | Artificially compresses feature variance and shifts the distribution of marginalized subgroups. |
| **BUG-006** | Currency Syntax Failure | ChatGPT (Zero-Shot) | **Medium (P2)** | Blind `replace('$', '')` failing on `'k'` multipliers | Values such as `'75k'` or `'120k'` fail conversion and silently coerce to `NaN`. |
| **BUG-007** | Categorical Misclassification | Gemini (Engineered) | **Low (P3)** | Incomplete country alias mapping table | International synonyms (`"Deutschland"`, `"Oz"`) unmapped and default to `"Other"`. |
| **BUG-008** | Incomplete Deduplication | Both AI Models | **Medium (P2)** | Key deduplication executed prior to whitespace trimming | Records with `' CUST-1042 '` vs `'CUST-1042'` persist as separate duplicate rows. |

---

## Detailed Failure Mode Analysis

### BUG-001: The Silent `dropna()` Catastrophe
* **Severity**: Critical (P0)
* **Model**: ChatGPT Zero-Shot
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df.dropna(inplace=True)
  ```
* **Mechanism of Failure**:
  In real-world data, missing values rarely occur uniformly across columns. In our benchmark dataset:
  * `phone_number` contains 38% missingness (Missing Completely At Random).
  * `annual_income` contains 30% missingness.
  * `satisfaction_score` contains 20% missingness.
  When an AI places `df.dropna()` without specifying `subset=['customer_id']`, pandas drops any row having *at least one* missing value. Consequently, 1,244 out of 1,260 rows (98.7%) were deleted without a warning or error message.
* **Engineering Remedy**:
  Always mandate key-based row dropping:
  ```python
  df = df.dropna(subset=['customer_id'])
  # Handle optional attributes using domain-appropriate defaults or flags
  df['phone_number'] = df['phone_number'].fillna('Not Provided')
  ```

---

### BUG-002: European Date Inversion & `NaT` Generation
* **Severity**: High (P1)
* **Model**: ChatGPT Zero-Shot
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df['signup_date'] = pd.to_datetime(df['signup_date'])
  ```
* **Mechanism of Failure**:
  By default, pandas assumes American format `MM/DD/YYYY`. When encountering `04/05/2023`, it parses the date as April 5th rather than May 4th. When encountering `15/01/2023`, where the first number exceeds 12, it either crashes or returns `NaT` if `errors='coerce'` is specified. This corrupts temporal cohorts and time-series aggregations.
* **Engineering Remedy**:
  Employ cascading multi-format parsing with explicit international defaults:
  ```python
  df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce', dayfirst=True)
  ```

---

### BUG-003: Text Mutilation via Overzealous Regex
* **Severity**: Medium (P2)
* **Model**: ChatGPT Zero-Shot
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df['full_name'] = df['full_name'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
  ```
* **Mechanism of Failure**:
  The AI assumes that any character that is not a Latin letter or space represents dirt or noise. This regex silently strips:
  * Apostrophes: `O'Connor` &rarr; `OConnor`, `D'Angelo` &rarr; `DAngelo`
  * Hyphens: `Jean-Luc` &rarr; `JeanLuc`, `Mary-Jane` &rarr; `MaryJane`
  * Accents: `Renée` &rarr; `Rene`
  This degrades identity matching across CRM systems and introduces customer name errors.
* **Engineering Remedy**:
  Preserve culturally valid punctuation and normalize title casing around separators:
  ```python
  def safe_clean_name(name):
      if pd.isna(name): return np.nan
      s = re.sub(r'^(Dr\.|Mr\.|Ms\.|Prof\.)\s+', '', str(name).strip(), flags=re.I)
      return re.sub(r'\s+', ' ', s)
  ```

---

### BUG-004: Inversion of Financial Accounting Logic
* **Severity**: Critical (P0)
* **Model**: ChatGPT Zero-Shot
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df['purchase_amount'] = df['purchase_amount'].abs()
  ```
* **Mechanism of Failure**:
  The AI treats negative numbers in transaction amounts as erroneous input, stripping the negative sign with `.abs()`. In retail and e-commerce schemas, negative values represent legitimate returns, refunds, or chargebacks. Converting `-$85.00` to `+$85.00` results in a **$170 accounting swing per return**, distorting revenue, customer lifetime value (LTV), and churn indicators.
* **Engineering Remedy**:
  Distinguish sign semantics from currency formatting:
  ```python
  # Preserve negative signs while removing currency symbols
  is_neg = s.startswith('-') or (s.startswith('(') and s.endswith(')'))
  clean_num = float(re.sub(r'[^\d.]', '', s))
  final_val = -clean_num if is_neg else clean_num
  ```

---

### BUG-005: Responsible AI Violation — Subgroup Distortion via Global Mean Imputation
* **Severity**: High (P1)
* **Model**: ChatGPT & Gemini Flash
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df['annual_income'] = df['annual_income'].fillna(df['annual_income'].mean())
  ```
* **Mechanism of Failure**:
  Replacing all missing salaries with a single global mean ($64,500):
  1. Collapses the natural standard deviation of the feature by over 18%.
  2. Creates an artificial spike at the exact mean value in histogram distributions.
  3. Violates demographic fairness: junior demographics (e.g. students or 18-24 age bracket) are artificially inflated to the global average, while high-income demographics are depressed.
* **Engineering Remedy**:
  Implement subgroup-aware conditional imputation (e.g. median income conditioned on `Country` and `Age Group`) and append an imputation indicator flag (`is_annual_income_imputed`) to preserve statistical lineage.

---

### BUG-006: Multiplier Ignorance in String Parsing
* **Severity**: Medium (P2)
* **Model**: ChatGPT Zero-Shot
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df['annual_income'] = pd.to_numeric(df['annual_income'].str.replace('$', ''))
  ```
* **Mechanism of Failure**:
  Many public datasets and user inputs represent salaries using human shorthand, such as `'75k'` or `'120k'`. The simple `.replace('$', '')` leaves the `'k'` character intact, causing `pd.to_numeric(..., errors='coerce')` to convert the value into `NaN`.
* **Engineering Remedy**:
  Detect multiplier units explicitly:
  ```python
  if 'k' in s:
      val = float(s.replace('k', '')) * 1000.0
  ```

---

### BUG-007: Out-of-Vocabulary Categorical Aliases
* **Severity**: Low (P3)
* **Model**: Gemini Prompt-Engineered
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  country_lookup = {'us': 'United States', 'usa': 'United States', 'uk': 'United Kingdom'}
  df['country'] = df['country'].str.lower().map(country_lookup).fillna('Other')
  ```
* **Mechanism of Failure**:
  While prompt-engineered code includes basic dictionary lookups, it inevitably misses regional synonyms such as `"Deutschland"` for Germany or colloquial abbreviations like `"Oz"` for Australia. As a result, valid entries are mapped to `"Other"`.
* **Engineering Remedy**:
  Use exhaustive ISO-standard lookups or fuzzy string matching (`rapidfuzz` / canonical geographic dictionaries).

---

### BUG-008: Premature Deduplication Prior to String Normalization
* **Severity**: Medium (P2)
* **Model**: Both AI Models
* **Code Excerpt**:
  ```python
  # AI-generated snippet
  df.drop_duplicates(subset=['customer_id'], inplace=True)
  ```
* **Mechanism of Failure**:
  In raw datasets, duplicate records often feature whitespace discrepancies (e.g. `' CUST-1042 '` vs `'CUST-1042'`) or case variations (`'cust-1042'`). Executing deduplication prior to whitespace trimming allows duplicate records to slip through undetected.
* **Engineering Remedy**:
  Normalize keys *before* deduplication:
  ```python
  df['customer_id'] = df['customer_id'].astype(str).str.strip().str.upper()
  df = df.drop_duplicates(subset=['customer_id'], keep='first')
  ```

---

## Conclusion & Practitioner Recommendations
The empirical findings demonstrate that while LLMs generate clean-looking, syntactically valid pandas code in seconds, **their default outputs are inherently untrustworthy for mission-critical data pipelines**. 

To safely leverage AI in data cleaning workflows:
1. **Never deploy zero-shot AI code without human review.**
2. **Explicitly forbid global `dropna()` in system prompts.**
3. **Always supply an explicit data dictionary with format definitions (e.g. DD/MM/YYYY, negative refund semantics).**
4. **Wrap AI-generated cleaning routines in automated unit tests and statistical drift checks before committing to production.**
