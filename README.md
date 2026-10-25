# CAPSTONE 65 — AI Data Cleaning Assistant with Python

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ### **Core Project Question:**
> **"How reliable is AI-generated data-cleaning code?"**
>
> **Executive Answer:**  
> AI-generated data-cleaning code is **unreliable when generated in zero-shot mode (reliability score: 10.0/100)** due to catastrophic failure modes such as blind `df.dropna(inplace=True)` (erasing 98.7% of valid data), chronological date swapping, and erasure of financial refunds. However, when guided by **prompt-engineered schema contracts (score: 91.5/100)** or wrapped in a **defensive, self-healing, auditable Python pipeline (`AIDataCleaningAssistant`, score: 98.5/100)**, AI tools become formidable assistants that accelerate data preparation safely.

---

## 📋 Required AI Skills & Free-Tier Tools Matrix

### Demonstrated AI Skills (Topics 1–9)
| # | Skill Topic | How Demonstrated in Capstone 65 |
| :-: | :--- | :--- |
| 1 | **AI Foundations** | Empirical analysis of LLM stochastic generation, tokenization limits, and code hallucination patterns. |
| 2 | **Prompt Engineering** | Design and evaluation of 6 distinct prompt architectures (Zero-shot, Few-shot with Schema, Chain-of-Thought, Defensive Persona, Self-Correction, Productionizer). |
| 3 | **Research & Verification** | Formal hypothesis testing (two-sample Kolmogorov-Smirnov test, Wasserstein distance, schema validation). |
| 4 | **Automation** | Engineering an industrial-grade, object-oriented pipeline class (`AIDataCleaningAssistant`) with execution audit logging. |
| 5 | **Responsible AI** | Demographic fairness audit detecting and preventing the 100% erasure of Non-Binary customers caused by naive zero-shot cleaning. Subgroup-aware imputation. |
| 6 | **Content Writing & Documentation** | Executive reports, failure taxonomies, slide deck presentation, and comprehensive before/after analysis. |

### Required Free-Tier Tools Used
1. **Google Colab**: Interactive, reproducible Jupyter notebook (`notebooks/capstone_65_ai_data_cleaning.ipynb`).
2. **Kaggle**: Benchmark structure inspired by real-world messy public e-commerce datasets.
3. **ChatGPT (Free Tier / GPT-4o mini)**: Zero-shot data-cleaning code simulation and failure mode evaluation.
4. **Gemini (Free Tier)**: Prompt-engineered schema-constrained code evaluation.
5. **GitHub**: Modular, open-source repository architecture.
6. **Google Sheets / Pandas**: Exportable tabular CSV outputs for stakeholder auditing.

---

## 📦 Complete Submission Deliverables Directory

All required final deliverables are organized within this repository:

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **1. Original & Cleaned Data** | [`data/raw/`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/data/raw/) & [`data/cleaned/`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/data/cleaned/) | Raw messy dataset (1,260 rows) and 4 cleaned datasets (ChatGPT, Gemini, Pipeline, Ground Truth). |
| **2. Code** | [`src/`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/src/) & [`notebooks/`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/notebooks/) | Modular Python source code and Google Colab notebook (`capstone_65_ai_data_cleaning.ipynb`). |
| **3. Bug Log** | [`reports/bug_log.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/reports/bug_log.md) & [`reports/bug_log.csv`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/reports/bug_log.csv) | Detailed documentation of 8 failure modes, code snippets, and remedies. |
| **4. Comparison** | [`reports/comparison_report.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/reports/comparison_report.md) | Deep comparative evaluation across retention, semantics, speed, and reliability. |
| **5. Presentation** | [`presentation/index.html`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/presentation/index.html) & [`presentation/capstone_presentation.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/presentation/capstone_presentation.md) | Interactive Web presentation slide deck with animations and speaker notes. |
| **6. Prompt Library** | [`prompts/prompt_library.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/prompts/prompt_library.md) | 6 reusable, evaluated prompt templates from zero-shot to productionizer. |
| **7. Verification Report** | [`reports/verification_report.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/reports/verification_report.md) | Schema checks, statistical hypothesis testing (KS-test), and Responsible AI audit. |
| **8. Before/After Analysis** | [`reports/before_after_analysis.md`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/reports/before_after_analysis.md) | Exploratory profiling, missingness breakdown, and distribution shift tables. |
| **9. Final Demo App** | [`app.py`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/app.py) | Interactive Streamlit web application with live diffs, metrics, and bug explorers. |

---

## 🔬 Benchmark Summary Scorecard

```
+---------------------------------------------------------------------------------------------------------+
|                                    CAPSTONE 65 BENCHMARK SCORECARD                                      |
+-----------------------------+-------------------+--------------------+-------------------+--------------+
| Metric                      | ChatGPT Zero-Shot | Gemini Engineered  | Reusable Pipeline | Ground Truth |
+-----------------------------+-------------------+--------------------+-------------------+--------------+
| Total Rows Retained         | 16 (1.27%)        | 1,173 (93.10%)     | 1,173 (93.10%)    | 1,200        |
| Valid Duplicates Removed    | 45 (Incomplete)   | 71                 | 71                | 75           |
| Negative Refunds Preserved  | 0 (0.0% Erased)   | 83 (85.57%)        | 78 (80.41%)       | 97 (100.0%)  |
| Name Punctuation Integrity  | 0.0% (Corrupted)  | 96.83%             | 96.83%            | 100.0%       |
| ISO-8601 Date Conformance   | 100.0%            | 100.0%             | 100.0%            | 100.0%       |
| Income KS-Test p-value      | 0.5576*           | 0.0372             | 0.3445 (High)     | 1.0000       |
| Age Wasserstein Distance    | 3.36 yrs          | 1.04 yrs           | 1.04 yrs          | 0.00 yrs     |
| Audit Log & Lineage         | None              | None               | Complete JSON     | Complete     |
| Composite Reliability Score | 10.0 / 100 (FATAL)| 91.5 / 100         | 98.5 / 100        | 100.0 / 100  |
+-----------------------------+-------------------+--------------------+-------------------+--------------+
```

---

## 🚀 Quickstart & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Experiment Suite
Executes all 4 models, generates outputs in `data/cleaned/`, and computes verification metrics:
```bash
python -u src/run_experiments.py
python -u src/verifier.py
```

### 3. Launch the Interactive Web Demo (Streamlit)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to test live file cleaning, inspect before/after diffs, and explore bug alerts.

### 4. View the Interactive Presentation Deck
Simply open `presentation/index.html` in any modern web browser or run:
```powershell
start presentation/index.html
```
Press the `Right Arrow` or `Spacebar` to advance slides. Click **"Show Speaker Notes"** to view presenter narration.

### 5. Run the Google Colab Notebook
Open [`notebooks/capstone_65_ai_data_cleaning.ipynb`](file:///c:/Users/GN%20Systems/Desktop/ai%20data%20clean%20with%20python/notebooks/capstone_65_ai_data_cleaning.ipynb) in VS Code, JupyterLab, or upload to [Google Colab](https://colab.research.google.com).

---

## 💡 The 3 Golden Rules for AI-Assisted Data Cleaning

1. **Rule 1: Never Use Zero-Shot Prompts for Data Pipelines**  
   Always provide an explicit schema contract. Explicitly forbid global `dropna()` and specify international date formats (`dayfirst=True`).

2. **Rule 2: AI Has No Domain Intuition — Explicitly Define Semantics**  
   Clarify whether negative values represent legitimate accounting returns or errors. Clarify whether null contact fields are optional.

3. **Rule 3: Enforce Automated Verification & Data Lineage**  
   Wrap AI-generated cleaning routines in pre/post retention assertions and statistical distribution tests before committing data to downstream analytics.
