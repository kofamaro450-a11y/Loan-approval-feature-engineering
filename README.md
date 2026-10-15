# Loan-approval-feature-engineering
Feature engineering and data preprocessing pipeline for loan approval prediction. Handles missing values, outliers, encoding, scaling and feature selection to produce an ML-ready dataset.
# Loan Approval Prediction: Feature Engineering & Data Preprocessing

**AnalystLab Africa Data analytics Internship**

Prepares the Loan Prediction dataset (home-loan applications) for a machine-learning model that predicts whether a loan should be approved (`Loan_Status`).

## Highlights

- 614 labelled applications, 11 input features, target 68.7% approved / 31.3% rejected
- Missing values handled with mode/median imputation plus missing-value flags (no rows dropped)
- Engineered features: total income, co-applicant flag, EMI, EMI-to-income, loan-to-income
- Outliers handled with `log1p` + IQR capping (skew of applicant income 6.5 → 0.5)
- Encoding: binary mapping, one-hot for `property_area`, ordinal `dependents`
- Scaling: `StandardScaler` on continuous features
- Feature selection: correlation heatmap, Random Forest importance, mutual information → 17 features kept
- Strongest signal: `credit_history` (correlation with approval 0.54)

## Repository Structure

```
├── README.md
├── notebook/
│   └── Week2_Loan_Feature_Engineering.ipynb   # full, executed workflow
├── data/
│   ├── train.csv                 # original labelled data
│   ├── test.csv                  # original unlabelled data
│   ├── loan_cleaned.csv          # cleaned + engineered, human-readable (614 x 19)
│   ├── loan_ml_ready.csv         # encoded, scaled, selected (614 x 18, target last)
│   └── loan_ml_ready_test.csv    # same pipeline applied to test file (367 x 18)
├── reports/
│   ├── Business_Understanding_Report.docx
│   └── Data_Preprocessing_Report.docx
└── images/                       # 9 figures used in the notebook and reports
```

## Pipeline Summary

| Step | Technique |
|---|---|
| Duplicates | Checked rows and `Loan_ID` (none found) |
| Missing values | Mode (categorical, credit history, term), median (loan amount), flag columns |
| Data types | `3+` → 3, floats → ints |
| Feature engineering | `total_income`, `has_coapplicant`, `emi`, `emi_to_income`, `loan_to_income` |
| Outliers | Boxplot, IQR and Z-score detection; `log1p` + IQR capping |
| Encoding | Label/binary, one-hot (drop first), ordinal |
| Scaling | `StandardScaler` (fit on train only) |
| Selection | Drop one of each pair with \|r\| > 0.80 (lower RF importance) |

## Run It

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
cd notebook
jupyter notebook Week2_Loan_Feature_Engineering.ipynb
```

The notebook reads `../data/train.csv` and `../data/test.csv` and regenerates all outputs.

## Data Source

[Loan Prediction Problem Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)


#AnalystLabAfrica
