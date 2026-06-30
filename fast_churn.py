"""
Fast Customer Churn Prediction - Optimized for F1 Score
========================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
TRAIN_PATH   = "training_dataset.csv"
TEST_PATH    = "testing_dataset.csv"
OUTPUT_PATH  = "submission.csv"
RANDOM_STATE = 42
N_FOLDS      = 5


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def engineer_features(df):
    """Enhanced feature engineering."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Date features
    if 'date_of_registration' in df.columns:
        dor = pd.to_datetime(df['date_of_registration'], errors='coerce')
        df['reg_year']      = dor.dt.year
        df['reg_month']     = dor.dt.month
        df['reg_quarter']   = dor.dt.quarter
        df['reg_dayofweek'] = dor.dt.dayofweek
        df['reg_is_weekend']= (dor.dt.dayofweek >= 5).astype(int)
        df.drop(columns=['date_of_registration'], inplace=True)

    # Tenure features
    tenure = df['customer_tenure_days'].clip(lower=1)
    df['is_new_customer'] = (tenure < 90).astype(int)
    df['is_churn_risky']  = ((tenure >= 30) & (tenure < 180)).astype(int)
    df['is_loyal']        = (tenure >= 365).astype(int)

    # Usage features
    df['total_usage']     = df['calls_made'] + df['sms_sent'] + df['data_used']
    df['call_share']      = df['calls_made'] / (df['total_usage'] + 1)
    df['sms_share']       = df['sms_sent'] / (df['total_usage'] + 1)
    df['data_share']      = df['data_used'] / (df['total_usage'] + 1)
    
    df['daily_engagement']= df['calls_per_day'] + df['sms_per_day'] + df['data_per_day']
    df['data_per_call']   = df['data_per_day'] / (df['calls_per_day'] + 1)
    df['call_intensity']  = df['calls_per_day'] * df['calls_made']

    # Financial interaction
    df['income_per_usage']= df['estimated_salary'] / (df['total_usage'] + 1)
    df['income_per_tenure']= df['estimated_salary'] / (tenure + 1)
    
    # Age features
    df['age_group']       = pd.cut(df['age'], bins=[0,25,35,50,65,120], labels=[0,1,2,3,4]).astype(int)
    df['age_tenure_score']= df['age'] * tenure / 100

    # Engagement flags
    df['high_usage']      = (df['total_usage'] > df['total_usage'].quantile(0.66)).astype(int)
    df['low_engagement']  = (df['activity_score'] < df['activity_score'].quantile(0.33)).astype(int)
    
    # Interaction terms
    df['tenure_x_activity'] = tenure * df['activity_score']
    df['age_x_usage']       = df['age'] * df['daily_engagement']

    # Log transforms
    for col in ['estimated_salary', 'calls_made', 'data_used', 'customer_tenure_days']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    # Polynomial features
    df['tenure_sq']       = tenure ** 2
    df['activity_sq']     = df['activity_score'] ** 2
    df['age_sq']          = df['age'] ** 2

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def preprocess_data(train_df, test_df):
    """Encode categorical variables."""
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        if col in train_df.columns:
            le = LabelEncoder()
            combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
            le.fit(combined)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col]  = le.transform(test_df[col].astype(str))

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols]  = imputer.transform(test_df[numeric_cols])

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
def find_best_threshold(y_true, y_proba):
    """Find best F1 threshold."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
def train_models(X, y, X_test):
    """Train ensemble with OOF stacking."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Initialize OOF arrays
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))
    
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    
    cat_oof = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    print("\nTraining ensemble models...")
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/{N_FOLDS}")
        
        Xtr, Xval = X.iloc[tr_idx].values, X.iloc[val_idx].values
        ytr, yval = y.iloc[tr_idx].values, y.iloc[val_idx].values

        # XGBoost
        print(f"    XGBoost...", end=' ')
        xgb_model = xgb.XGBClassifier(
            max_depth=7, learning_rate=0.05, n_estimators=600,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0, eval_metric='logloss'
        )
        xgb_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        xgb_oof[val_idx] = xgb_model.predict_proba(Xval)[:, 1]
        xgb_test += xgb_model.predict_proba(X_test.values)[:, 1] / N_FOLDS
        print(f"F1={f1_score(yval, (xgb_oof[val_idx] >= 0.5).astype(int)):.4f}")

        # LightGBM
        print(f"    LightGBM...", end=' ')
        lgb_model = lgb.LGBMClassifier(
            max_depth=8, learning_rate=0.05, n_estimators=600,
            num_leaves=127, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1)
        ])
        lgb_oof[val_idx] = lgb_model.predict_proba(Xval)[:, 1]
        lgb_test += lgb_model.predict_proba(X_test.values)[:, 1] / N_FOLDS
        print(f"F1={f1_score(yval, (lgb_oof[val_idx] >= 0.5).astype(int)):.4f}")

        # CatBoost
        print(f"    CatBoost...", end=' ')
        cat_model = CatBoostClassifier(
            depth=8, learning_rate=0.05, iterations=600, l2_leaf_reg=3,
            random_seed=RANDOM_STATE, verbose=False, auto_class_weights='Balanced'
        )
        cat_model.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=100, verbose=False)
        cat_oof[val_idx] = cat_model.predict_proba(Xval)[:, 1]
        cat_test += cat_model.predict_proba(X_test.values)[:, 1] / N_FOLDS
        print(f"F1={f1_score(yval, (cat_oof[val_idx] >= 0.5).astype(int)):.4f}")

    return (xgb_oof, xgb_test), (lgb_oof, lgb_test), (cat_oof, cat_test)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ADVANCED CHURN PREDICTION - FAST VERSION")
print("="*60)

# Load
print("\n[1/5] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"  Train: {train.shape} | Test: {test.shape}")
print(f"  Churn rate: {train['churn'].mean():.2%}")

# Feature Engineering
print("\n[2/5] Feature engineering...")
y_train = train['churn'].copy()
X_train = engineer_features(train.drop(columns=['churn']))
X_test = engineer_features(test)
print(f"  Features: {X_train.shape[1]}")

# Preprocessing
print("\n[3/5] Preprocessing...")
X_train, X_test = preprocess_data(X_train, X_test)

# Train
print("\n[4/5] Training models...")
(xgb_oof, xgb_test), (lgb_oof, lgb_test), (cat_oof, cat_test) = train_models(X_train, y_train, X_test)

# Ensemble
print("\n[5/5] Finalizing predictions...")
ensemble_oof = (xgb_oof + lgb_oof + cat_oof) / 3
ensemble_test = (xgb_test + lgb_test + cat_test) / 3

best_t, best_f1 = find_best_threshold(y_train.values, ensemble_oof)
print(f"\n  Best threshold: {best_t:.3f}")
print(f"  OOF F1 Score: {best_f1:.4f}")

# Submit
final_preds = (ensemble_test >= best_t).astype(int)
submission = pd.DataFrame({
    'id': np.arange(1, len(test) + 1),
    'churn': final_preds
})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Submission saved: {OUTPUT_PATH}")
print(f"  Predicted churned: {final_preds.sum()} ({final_preds.mean():.2%})")
print("\n" + "="*60)
print("COMPLETE!")
print("="*60 + "\n")
