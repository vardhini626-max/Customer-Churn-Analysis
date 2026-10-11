# Customer Churn Prediction (Telecom / Subscription)

Predict whether a telecom or subscription customer will **cancel their service** using usage and billing history. Everything lives in one Jupyter notebook: [`churn_prediction.ipynb`](churn_prediction.ipynb). It compares **Logistic Regression** and a **Decision Tree** using **Accuracy, Precision, Recall, F1 and ROC-AUC**.

## Skills demonstrated
- Logistic Regression and Decision Trees (scikit-learn)
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
- Preprocessing pipelines (imputation, scaling, one-hot encoding) with `ColumnTransformer`
- Hyperparameter tuning with stratified k-fold `GridSearchCV`
- Handling class imbalance (`class_weight="balanced"`)
- Model interpretation (coefficients, feature importance, tree plot) and decision-threshold tuning

## Project structure
```
customer-churn-prediction/
├── churn_prediction.ipynb     # full analysis (outputs included)
├── data/telecom_churn.csv     # synthetic dataset, 7,000 customers
├── reports/
│   ├── model_comparison.csv
│   └── figures/               # charts exported by the notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run
```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```
Then choose **Kernel → Restart & Run All**.

## Dataset
The data is **synthetic** but mimics a telecom provider: tenure, monthly and total charges, data usage, support calls, late payments, contract type, internet service, payment method, tech support, paperless billing and a senior-citizen flag. The target `churn` is 1 if the customer cancelled (about 34% of customers). About 1% of `total_charges` values are missing to practise imputation.

To use real data (for example the public *Telco Customer Churn* dataset), save it as `data/telecom_churn.csv` with a binary `churn` column and update the `NUMERIC` and `CATEGORICAL` lists in the notebook.

## Method
1. Stratified 80/20 train/test split.
2. Preprocessing: median imputation and scaling for numeric features; mode imputation and one-hot encoding for categorical features.
3. Models tuned with 5-fold stratified cross-validation (ROC-AUC): Logistic Regression (`C`) and Decision Tree (`max_depth`, `min_samples_leaf`).
4. Final evaluation on the held-out test set.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.731 | 0.581 | 0.719 | 0.643 | 0.794 |
| Decision Tree | 0.698 | 0.538 | 0.709 | 0.612 | 0.766 |

![ROC curves](reports/figures/roc_curves.png)
![Confusion matrices](reports/figures/confusion_matrices.png)
![LR coefficients](reports/figures/lr_coefficients.png)

## Key takeaways
- **Recall matters most for churn.** Missing a customer who is about to leave costs more than offering a retention discount to someone who would have stayed, so both models use balanced class weights.
- **Accuracy alone is misleading** on imbalanced data, so precision, recall, F1 and ROC-AUC are reported too.
- **Logistic Regression** performed best here and is easy to explain through its coefficients.
- **Decision Tree** gives human-readable rules but is slightly less accurate.
- Typical churn drivers: month-to-month contracts, short tenure, frequent support calls, late payments and electronic-check payment.

## Ideas for improvement
- Try Random Forest or Gradient Boosting and compare against these baselines
- Add SHAP explanations
- Tune the threshold to hit a target recall
- Wrap the model in a Streamlit app

## Author
YOUR NAME HERE, https://github.com/YOUR-USERNAME

## License
MIT
