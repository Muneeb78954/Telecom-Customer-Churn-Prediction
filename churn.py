# ==========================================
# CHURN PREDICTION - F1 SCORE OPTIMIZED
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier

# ==========================================
# LOAD DATA
# ==========================================

TRAIN_PATH = "training_dataset.csv"
TEST_PATH = "testing_dataset.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train Shape: {train.shape}")
print(f"Test Shape: {test.shape}")

# ==========================================
# SAVE IDS
# ==========================================

test_ids = test["customer_id"]

# ==========================================
# COMBINE DATA FOR CONSISTENT CLEANING
# ==========================================

target = "churn"

train_len = len(train)

combined = pd.concat(
    [
        train.drop(columns=[target]),
        test
    ],
    axis=0,
    ignore_index=True
)

# ==========================================
# REPLACE INFINITE VALUES
# ==========================================

combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================
# HANDLE NEGATIVE VALUES
# ==========================================

invalid_negative_cols = [
    'calls_made',
    'sms_sent',
    'data_used',
    'calls_per_day',
    'sms_per_day',
    'data_per_day',
    'salary_data_ratio',
    'activity_score'
]

for col in invalid_negative_cols:
    if col in combined.columns:
        combined.loc[combined[col] < 0, col] = np.nan

# ==========================================
# DATE FEATURES
# ==========================================

if "date_of_registration" in combined.columns:

    combined["date_of_registration"] = pd.to_datetime(
        combined["date_of_registration"],
        errors="coerce"
    )

    combined["registration_year"] = (
        combined["date_of_registration"].dt.year
    )

    combined["registration_month"] = (
        combined["date_of_registration"].dt.month
    )

    combined["registration_day"] = (
        combined["date_of_registration"].dt.day
    )

    combined["registration_weekday"] = (
        combined["date_of_registration"].dt.weekday
    )

    combined.drop(
        columns=["date_of_registration"],
        inplace=True
    )

# ==========================================
# FEATURE ENGINEERING
# ==========================================

if all(col in combined.columns for col in
       ["calls_per_day", "sms_per_day", "data_per_day"]):

    combined["engagement_ratio"] = (
        combined["calls_per_day"] +
        combined["sms_per_day"] +
        combined["data_per_day"]
    )

if all(col in combined.columns for col in
       ["estimated_salary", "calls_made"]):

    combined["salary_per_call"] = (
        combined["estimated_salary"] /
        (combined["calls_made"] + 1)
    )

if all(col in combined.columns for col in
       ["estimated_salary", "sms_sent"]):

    combined["salary_per_sms"] = (
        combined["estimated_salary"] /
        (combined["sms_sent"] + 1)
    )

if all(col in combined.columns for col in
       ["estimated_salary", "data_used"]):

    combined["salary_per_data"] = (
        combined["estimated_salary"] /
        (combined["data_used"] + 1)
    )

# ==========================================
# CATEGORICAL FEATURES
# ==========================================

cat_features = []

possible_cat_cols = [
    "telecom_partner",
    "gender",
    "state",
    "city"
]

for col in possible_cat_cols:
    if col in combined.columns:
        cat_features.append(col)

# ==========================================
# NUMERIC IMPUTATION
# ==========================================

numeric_cols = combined.select_dtypes(
    include=["int64", "float64"]
).columns

imputer = SimpleImputer(strategy="median")

combined[numeric_cols] = imputer.fit_transform(
    combined[numeric_cols]
)

# ==========================================
# CATEGORICAL IMPUTATION
# ==========================================

for col in cat_features:
    combined[col] = combined[col].fillna("Missing")

# ==========================================
# SPLIT BACK
# ==========================================

X_train = combined.iloc[:train_len].copy()
X_test = combined.iloc[train_len:].copy()

y_train = train[target]

# ==========================================
# CROSS VALIDATION
# ==========================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X_train, y_train)):

    print(f"\nFold {fold+1}")

    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]

    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='F1',
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    oof_pred[val_idx] = model.predict_proba(
        X_val
    )[:, 1]

    test_pred += (
        model.predict_proba(X_test)[:, 1]
        / skf.n_splits
    )

# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================

best_threshold = 0.50
best_f1 = 0

for threshold in np.arange(0.20, 0.81, 0.01):

    preds = (
        oof_pred >= threshold
    ).astype(int)

    score = f1_score(
        y_train,
        preds
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\n======================")
print("BEST F1:", best_f1)
print("BEST THRESHOLD:", best_threshold)
print("======================")

# ==========================================
# FINAL PREDICTIONS
# ==========================================

final_preds = (
    test_pred >= best_threshold
).astype(int)

# ==========================================
# SUBMISSION FILE
# ==========================================

submission = pd.DataFrame({
    "customer_id": test_ids,
    "churn": final_preds
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully!")
print(submission.head())