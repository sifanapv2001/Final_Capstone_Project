# TESS Exoplanet Detection and Classification

## Primary objective
Binary classification of TESS Objects of Interest into:
- 1 = planetary/planet-candidate side (PC, CP, KP)
- 0 = false-signal side (FP, FA)

APC is excluded from the strict binary experiment because it is ambiguous.

## Secondary objective
Multiclass classification using the original `tfopwg_disp` labels:
PC, FP, CP, KP, APC, FA.

## Pipeline
Data extraction → cleaning → EDA → feature engineering → transformation → train/test split → stratified CV → ML models → hyperparameter tuning → ANN → SHAP → model selection → deployment preparation.

## Models
Logistic Regression, Random Forest, XGBoost, SVM, ANN.

## Metrics
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.

## Files
- `NASA_Exoplanet_Binary_Multiclass_Project.ipynb`
- `NASA_Exoplanet_Project_Report.docx`

## Data
Place the supplied CSV in the notebook working directory or update `DATA_PATH`.
