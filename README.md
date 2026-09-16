# Customer-Churn-Prediction-Model

Predicts customer churn on a telecom dataset (7,000+ records, 26.5% churn rate), comparing multiple classification models and handling class imbalance to improve recall on churned customers.

**Dataset**

Telco customer data with 7,000+ records and 26.5% churn rate. Features include tenure, contract type, monthly charges, and service subscriptions.

**Approach**
Cleaned and preprocessed data (type conversions, missing values, feature encoding)
Built and compared Decision Tree, Random Forest, XGBoost, and AdaBoost classifiers
Addressed class imbalance using SMOTE and class-weighting
Tuned hyperparameters with RandomizedSearchCV
Results

**Final model (class-weighted):** 75% accuracy, 76% recall on churned customers.

**Files**
CustomerChurn — full analysis and model building
Customer-Churn.csv — dataset

**Tools**
Python, Pandas, Scikit-learn, XGBoost
