# Loan Limit Optimization (ML + OR)

**Author:** Christopher Nguu

> **Data disclosure:** All borrower-, offer-, portfolio-, and performance-level data in this repository are **synthetic**. They are illustrative research data, not production records or proprietary lender data.

Originally developed as a take-home-style case study for a senior data science hiring assessment in consumer lending / credit decisioning; reframed here as a reproducible MSc Data Science & Analytics portfolio project.

## Overview

This project demonstrates a transparent predict-then-optimize workflow for increasing consumer credit limits. It combines probability-of-default (PD), behavioral response models, portfolio simulation, and constrained operations research (OR).

**Methods:** PD/behavior ML + constrained OR for limit-increase tiers + macro stress.

The optimization allocates discrete limit-increase tiers across risk cohorts while respecting incremental-budget, expected-loss, concentration, and stress-loss constraints. Results are illustrative and should not be used for real lending decisions without validated data, governance, monitoring, and compliance review.

## How to run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the optimization simulation and generate figures/data
python loan_limit_engine.py

# Run out-of-time-style behavioral model validation
python ml_behavioral_models.py

# Execute the notebook (optional)
jupyter nbconvert --to notebook --execute loan_limit_optimization.ipynb
```

The scripts write reproducible outputs to `data/` and `figures/`. The checked-in report is available in Markdown and PDF.

## What's in the repo

- `loan_limit_optimization.ipynb` — end-to-end walkthrough.
- `loan_limit_engine.py` — synthetic portfolio generation, scenario simulation, and constrained MILP allocation.
- `ml_behavioral_models.py` — PD, acceptance, and utilization modeling.
- `build_report_pdf.py` — report-generation helper.
- `data/` — synthetic borrower, offer, cohort, metrics, and optimization outputs.
- `figures/` — model-validation, portfolio, optimization, and stress-test visuals.
- `Loan_Limit_Optimization_Report.md` / `.pdf` — project report.
- `requirements.txt` — Python dependencies.

## Scope and limitations

The assumptions, labels, elasticities, and macro scenarios are illustrative. The results are not evidence of model performance on any real lender's portfolio. Production use would require representative historical data, independent validation, fairness and explainability review, credit-policy controls, and ongoing monitoring.
