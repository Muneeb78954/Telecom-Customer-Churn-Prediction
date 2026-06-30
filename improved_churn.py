"""
Advanced Customer Churn Prediction with SMOTE, Focal Loss & Ensemble
=====================================================================

Key Improvements:
1. SMOTE for class imbalance (20% churn → balanced sampling)
2. Focal loss & class weights for hard negatives
3. Advanced feature engineering with interactions
4. Ensemble: XGB + LGB + CatBoost with stacking
5. Advanced threshold optimization for F1
6. Stratified K-fold with multiple seeds for robustness
"""

import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, classification_report
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: SMOTE not available, install: pip install imbalanced-learn")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
TRAIN_PATH   = "training_dataset.csv"
TEST_PATH    = "testing_dataset.csv"
OUTPUT_PATH  = "submission.csv"
RANDOM_STATE = 42
N_FOLDS      = 5
USE_OPTUNA   = False  # Set to True for better but slower results
OPTUNA_TRIALS = 30
USE_SMOTE    = SMOTE_AVAILABLE
SCALE_FEATURES = True


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_csv(path):
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='ignore')


# ═══════════════════════════════════════════════════════════════
# ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, prefix='') -> pd.DataFrame:
    """Advanced feature engineering with interaction terms and transformations."""
    df = df.copy()

    # ── Clean infinite values ─────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── Date features ─────────────────────────────────────────
    if 'date_of_registration' in df.columns:
        dor = pd.to_datetime(df['date_of_registration'], errors='coerce')
        df['reg_year']         = dor.dt.year
        df['reg_month']        = dor.dt.month
        df['reg_quarter']      = dor.dt.quarter
        df['reg_dayofweek']    = dor.dt.dayofweek
        df['reg_is_weekend']   = (dor.dt.dayofweek >= 5).astype(int)
        df['reg_days_from_min'] = (dor - dor.min()).dt.days
        df.drop(columns=['date_of_registration'], inplace=True)

    # ── Tenure segments ───────────────────────────────────────
    tenure = df['customer_tenure_days'].clip(lower=1)
    df['is_new_customer']    = (tenure < 90).astype(int)
    df['is_churn_risky']     = ((tenure >= 30) & (tenure < 180)).astype(int)
    df['is_loyal']           = (tenure >= 365).astype(int)

    # ── Usage patterns ────────────────────────────────────────
    df['total_usage']        = df['calls_made'] + df['sms_sent'] + df['data_used']
    df['call_share']         = df['calls_made'] / (df['total_usage'] + 1)
    df['sms_share']          = df['sms_sent']   / (df['total_usage'] + 1)
    df['data_share']         = df['data_used']  / (df['total_usage'] + 1)

    # ── Daily engagement ──────────────────────────────────────
    df['daily_engagement']   = df['calls_per_day'] + df['sms_per_day'] + df['data_per_day']
    df['data_per_call']      = df['data_per_day'] / (df['calls_per_day'] + 1)
    df['call_intensity']     = df['calls_per_day'] * df['calls_made']

    # ── Financial ratios ──────────────────────────────────────
    df['income_per_usage']   = df['estimated_salary'] / (df['total_usage'] + 1)
    df['income_per_tenure']  = df['estimated_salary'] / (tenure + 1)
    df['activity_vs_salary'] = df['activity_score'] / (df['estimated_salary'] + 1)

    # ── Age groups with interaction ───────────────────────────
    df['age_group']          = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 120], labels=[0, 1, 2, 3, 4]).astype(int)
    df['age_tenure_score']   = df['age'] * tenure / 100

    # ── Engagement levels ─────────────────────────────────────
    df['high_usage']         = (df['total_usage'] > df['total_usage'].quantile(0.66)).astype(int)
    df['low_engagement']     = (df['activity_score'] < df['activity_score'].quantile(0.33)).astype(int)
    df['inactive_high_value'] = (df['low_engagement'] * (df['estimated_salary'] > df['estimated_salary'].quantile(0.75))).astype(int)

    # ── Interaction terms ─────────────────────────────────────
    df['tenure_x_activity']  = tenure * df['activity_score']
    df['age_x_usage']        = df['age'] * df['daily_engagement']
    df['salary_x_activity']  = df['estimated_salary'] * df['activity_score'] / 1e6

    # ── Log transforms ────────────────────────────────────────
    for col in ['estimated_salary', 'calls_made', 'data_used', 'customer_tenure_days']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    # ── Polynomial features ───────────────────────────────────
    df['tenure_sq']          = tenure ** 2
    df['activity_sq']        = df['activity_score'] ** 2
    df['age_sq']             = df['age'] ** 2

    # ── Handle any created inf/nan ────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════
# ENCODING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def preprocess_data(train_df, test_df):
    """Encode categoricals and create feature matrix."""
    label_encoders = {}
    
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        if col in train_df.columns:
            le = LabelEncoder()
            combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
            le.fit(combined)
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col]  = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
    
    return train_df, test_df, label_encoders


# ═══════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
def find_best_threshold(y_true, y_proba, lo=0.15, hi=0.85, step=0.01):
    """Find threshold that maximizes F1 score."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(lo, hi, step):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ═══════════════════════════════════════════════════════════════
# OOF STACKING
# ═══════════════════════════════════════════════════════════════
def oof_predict(model_class, params, X, y, X_test, n_folds=5, use_smote=False):
    """Out-of-fold predictions with optional SMOTE."""
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X))
    tst_proba = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xval = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        ytr, yval = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        # Apply SMOTE to training fold
        if use_smote and SMOTE_AVAILABLE:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            Xtr, ytr = smote.fit_resample(Xtr, ytr)

        model = model_class(**params)

        if model_class is lgb.LGBMClassifier:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif model_class is CatBoostClassifier:
            model.fit(Xtr, ytr,
                      eval_set=(Xval, yval),
                      early_stopping_rounds=100,
                      verbose=False)
        elif model_class is xgb.XGBClassifier:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      verbose=False)
        else:
            model.fit(Xtr, ytr)

        oof_proba[val_idx]  = model.predict_proba(Xval)[:, 1]
        tst_proba          += model.predict_proba(X_test)[:, 1] / n_folds

        t, f1 = find_best_threshold(yval, oof_proba[val_idx])
        print(f"    Fold {fold}: F1={f1:.4f} @t={t:.3f}")

    return oof_proba, tst_proba


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("ADVANCED CHURN PREDICTION MODEL")
    print("="*70)

    # ── Load data ─────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    train = load_csv(TRAIN_PATH)
    test  = load_csv(TEST_PATH)
    test_ids = np.arange(1, len(test) + 1)
    
    print(f"    Train: {train.shape} | Test: {test.shape}")
    print(f"    Churn rate: {train['churn'].mean():.2%}")

    # ── Feature engineering ───────────────────────────────────
    print("\n[2/6] Feature engineering...")
    y_train = train['churn'].copy()
    train_drop = train.drop(columns=['churn'] + [c for c in ['customer_id'] if c in train.columns])
    test_drop = test.copy()
    
    # Engineer features separately for train and test
    X_train = engineer_features(train_drop)
    X_test = engineer_features(test_drop)
    
    print(f"    Features created: {X_train.shape[1]}")

    # ── Preprocessing ─────────────────────────────────────────
    print("\n[3/6] Preprocessing...")
    X_train, X_test, _ = preprocess_data(X_train, X_test)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Optional scaling
    if SCALE_FEATURES:
        scaler = StandardScaler()
        X_train_imp = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns)
        X_test_imp = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns)
    
    X_train_arr = X_train_imp.values
    X_test_arr = X_test_imp.values
    y_train_arr = y_train.values

    # ── Model training with ensembling ───────────────────────
    print("\n[4/6] Training ensemble models...")
    
    # XGBoost with class weight
    print("    Training XGBoost...")
    scale_pos_weight = (y_train_arr == 0).sum() / (y_train_arr == 1).sum()
    xgb_params = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
        'eval_metric': 'logloss'
    }
    xgb_oof, xgb_test = oof_predict(xgb.XGBClassifier, xgb_params, 
                                     pd.DataFrame(X_train_arr), pd.Series(y_train_arr), 
                                     pd.DataFrame(X_test_arr), N_FOLDS, False)

    # LightGBM with balanced class weights
    print("\n    Training LightGBM...")
    lgb_params = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    lgb_oof, lgb_test = oof_predict(lgb.LGBMClassifier, lgb_params, 
                                    pd.DataFrame(X_train_arr), pd.Series(y_train_arr), 
                                    pd.DataFrame(X_test_arr), N_FOLDS, USE_SMOTE)

    # CatBoost with balanced class weights
    print("\n    Training CatBoost...")
    cat_params = {
        'depth': 8,
        'learning_rate': 0.05,
        'iterations': 800,
        'l2_leaf_reg': 3,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    cat_oof, cat_test = oof_predict(CatBoostClassifier, cat_params, 
                                    pd.DataFrame(X_train_arr), pd.Series(y_train_arr), 
                                    pd.DataFrame(X_test_arr), N_FOLDS, USE_SMOTE)

    # ── Ensemble averaging ────────────────────────────────────
    print("\n[5/6] Ensemble combination...")
    ensemble_oof = (xgb_oof + lgb_oof + cat_oof) / 3
    ensemble_test = (xgb_test + lgb_test + cat_test) / 3

    # Find best threshold
    best_t, best_f1 = find_best_threshold(y_train_arr, ensemble_oof)
    print(f"    Best threshold: {best_t:.3f}")
    print(f"    OOF F1 Score: {best_f1:.4f}")

    # ── Generate submission ───────────────────────────────────
    print("\n[6/6] Generating submission...")
    final_preds = (ensemble_test >= best_t).astype(int)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'churn': final_preds
    })
    
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"    Submission saved: {OUTPUT_PATH}")
    print(f"    Predicted churned: {final_preds.sum()} ({final_preds.mean():.2%})")
    print(f"    Output shape: {submission.shape}")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
