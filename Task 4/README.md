🎯 Objective

Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

This task demonstrates:

    1)Data preprocessing (scaling, encoding)

    2)Pipeline API in Scikit-learn

    3)Hyperparameter tuning with GridSearchCV

    4)Model evaluation using accuracy & F1-score

    5)Exporting the pipeline for reusability in production

📂 Dataset

Telco Customer Churn Dataset
Available on Kaggle.

Features: Customer demographics, account details, service subscriptions.

Target: Churn (Yes → 1, No → 0)

🛠️ Steps Implemented
1. Data Preprocessing

    Dropped irrelevant columns (e.g., customerID)

    Encoded target variable (Yes/No → 1/0)

    Split dataset into train (80%) and test (20%)

2. Feature Engineering

    Identified categorical and numeric features

    Applied:

    StandardScaler → numeric features

    OneHotEncoder → categorical features

3. ML Pipelines

    Created two pipelines:

    Logistic Regression

    Random Forest

4. Hyperparameter Tuning

    Used GridSearchCV to tune parameters:

    Logistic Regression: C values

    Random Forest: n_estimators, max_depth

5. Evaluation

    Evaluated with Accuracy and F1-score

    Compared both models

    Chose the best model

6. Model Export

    Saved best pipeline as:

    churn_pipeline.pkl


Reloadable with:

    import joblib
    model = joblib.load("churn_pipeline.pkl")
    predictions = model.predict(new_data)

📊 Results

Logistic Regression: Accuracy ≈ 80%, F1 ≈ 58%

Random Forest: Accuracy ≈ 82%, F1 ≈ 54%

✅ Random Forest chosen as best model
