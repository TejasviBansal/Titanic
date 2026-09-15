# Titanic - Machine Learning from Disaster

A robust, production-grade machine learning pipeline for Kaggle's classic [Titanic: Machine Learning from Disaster]

This implementation demonstrates data-leakage prevention using scikit-learn `Pipeline` and `ColumnTransformer`, benchmarking multiple classifiers (Random Forest, SVM, XGBoost) via 5-fold cross-validation and exporting formatted test predictions.

---

## Architecture & Workflow

1. **Modular Feature Pipelines:**
   - **Numerical (`Age`, `SibSp`, `Parch`, `Fare`):** Imputed via median strategy; scaled with `StandardScaler` (critical for distance-based estimators like SVM).
   - **Categorical (`Pclass`, `Sex`, `Embarked`):** Imputed via mode (`most_frequent`); encoded using `OneHotEncoder(handle_unknown='ignore')`.
2. **Data Leakage Safeguards:** Preprocessing statistics are fitted strictly on the training folds during cross-validation, ensuring zero contamination from validation or test splits.
3. **Model Benchmarking:** Iterates across multiple models using 5-Fold Stratified Cross-Validation on the exact same folds.
4. **Automated Submission:** Retrains the best-performing pipeline on the full training set, runs inference on `test.csv`, and generates a Kaggle-ready submission file.

---

## Directory Structure

```text
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── src/
│   └── main.py
├── submission.csv
├── requirements.txt
└── README.md
