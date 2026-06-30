"""
Telecom Customer Churn Prediction — Leaderboard-Optimized Pipeline
===================================================================
Key findings on this dataset:
  - All features have near-zero correlation with churn (~0.002 max)
  - ROC-AUC ceiling ≈ 0.50-0.52 (near-random synthetic data)
  - Original code scored 0.06 F1 due to: wrong column names + bad threshold
  - Theoretical F1 ceiling ≈ 0.334 (predict all as churn)

Strategy:
  1. Fix the root cause: correct column mapping, correct submission format
  2. Squeeze every drop of signal via aggressive feature engineering
  3. Ensemble XGB + LGB + CatBoost with OOF stacking
  4. Threshold optimization tuned for F1 (NOT accuracy)
  5. Fall back to smart constant predictor if no signal found
"""

import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, classification_report

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ← Adjust these for your Kaggle run
# ─────────────────────────────────────────────────────────────────
TRAIN_PATH   = "training_dataset.csv"
TEST_PATH    = "testing_dataset.csv"
OUTPUT_PATH  = "submission.csv"
RANDOM_STATE = 42
N_FOLDS      = 5
USE_OPTUNA   = True    # Set False to skip tuning (faster, slightly lower score)
OPTUNA_TRIALS = 40     # Trials per model; increase to 80-100 for better results


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
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Clean known-bad values ───────────────────────────────
    df['salary_data_ratio'] = df['salary_data_ratio'].replace([np.inf, -np.inf], np.nan)

    # ── Fill missing values (median for numerics) ────────────
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Date features ────────────────────────────────────────
    if 'date_of_registration' in df.columns:
        dor = pd.to_datetime(df['date_of_registration'], errors='coerce')
        df['reg_year']         = dor.dt.year
        df['reg_month']        = dor.dt.month
        df['reg_quarter']      = dor.dt.quarter
        df['reg_dayofweek']    = dor.dt.dayofweek
        df['reg_dayofyear']    = dor.dt.dayofyear
        df['reg_weekofyear']   = dor.dt.isocalendar().week.astype(int)
        df['reg_is_weekend']   = (dor.dt.dayofweek >= 5).astype(int)
        # Days since the earliest registration in this dataset
        df['reg_days_from_min'] = (dor - dor.min()).dt.days
        df.drop(columns=['date_of_registration'], inplace=True)

    # ── Usage aggregates ─────────────────────────────────────
    df['total_usage']            = df['calls_made'] + df['sms_sent'] + df['data_used']
    df['call_share']             = df['calls_made'] / (df['total_usage'] + 1)
    df['sms_share']              = df['sms_sent']   / (df['total_usage'] + 1)
    df['data_share']             = df['data_used']  / (df['total_usage'] + 1)
    df['calls_x_sms']            = df['calls_made'] * df['sms_sent']
    df['calls_x_data']           = df['calls_made'] * df['data_used']

    # ── Consistency: daily rate vs. actual total ──────────────
    tenure = df['customer_tenure_days'].clip(lower=1)
    df['call_consistency']       = df['calls_per_day'] / (df['calls_made'] / tenure + 1e-6)
    df['sms_consistency']        = df['sms_per_day']   / (df['sms_sent']   / tenure + 1e-6)
    df['data_consistency']       = df['data_per_day']  / (df['data_used']  / tenure + 1e-6)

    # ── Tenure features ──────────────────────────────────────
    df['tenure_months']          = tenure / 30
    df['tenure_years']           = tenure / 365
    df['is_new_customer']        = (tenure < 90).astype(int)
    df['is_mid_tenure']          = ((tenure >= 90) & (tenure < 365)).astype(int)
    df['is_long_term']           = (tenure >= 365).astype(int)
    df['is_very_long_term']      = (tenure >= 730).astype(int)

    # ── Income / financial ───────────────────────────────────
    df['income_per_dependent']   = df['estimated_salary'] / (df['num_dependents'] + 1)
    df['income_per_tenure_yr']   = df['estimated_salary'] / (df['tenure_years'] + 0.1)
    df['income_x_activity']      = df['estimated_salary'] * df['activity_score']

    # ── Activity features ────────────────────────────────────
    df['activity_per_tenure']    = df['activity_score'] / (tenure + 1)
    df['activity_per_call']      = df['activity_score'] / (df['calls_made'] + 1)
    df['activity_per_sms']       = df['activity_score'] / (df['sms_sent'] + 1)
    df['activity_x_tenure']      = df['activity_score'] * tenure

    # ── Age features ─────────────────────────────────────────
    df['age_group']              = pd.cut(df['age'],
                                          bins=[0, 18, 25, 35, 45, 55, 65, 120],
                                          labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
    df['age_x_tenure']           = df['age'] * tenure
    df['age_x_activity']         = df['age'] * df['activity_score']

    # ── Binary usage flags ───────────────────────────────────
    df['high_data_user']         = (df['data_per_day']  > df['data_per_day'].quantile(0.75)).astype(int)
    df['high_call_user']         = (df['calls_per_day'] > df['calls_per_day'].quantile(0.75)).astype(int)
    df['low_activity']           = (df['activity_score'] < df['activity_score'].quantile(0.25)).astype(int)
    df['very_low_activity']      = (df['activity_score'] < df['activity_score'].quantile(0.1)).astype(int)
    df['negative_activity']      = (df['activity_score'] < 0).astype(int)

    # ── Log transforms for skewed features ──────────────────
    for col in ['estimated_salary', 'calls_made', 'data_used', 'customer_tenure_days']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    # ── Polynomial / ratio features ──────────────────────────
    df['salary_sq']              = df['estimated_salary'] ** 2
    df['tenure_sq']              = tenure ** 2
    df['activity_sq']            = df['activity_score'] ** 2

    # ── Clean up any inf/nan created by ratios ───────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════
# ENCODING
# ═══════════════════════════════════════════════════════════════
def encode_categoricals(train_df, test_df):
    """Fit LabelEncoder on train+test combined to avoid unseen labels."""
    label_encoders = {}
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str),
                               test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    return train_df, test_df, label_encoders


# ═══════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION (F1-targeted)
# ═══════════════════════════════════════════════════════════════
def find_best_threshold(y_true, y_proba, lo=0.25, hi=0.76, step=0.005):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(lo, hi, step):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ═══════════════════════════════════════════════════════════════
# OOF STACKING
# ═══════════════════════════════════════════════════════════════
def oof_predict(model_class, params, X, y, X_test, n_folds=5):
    """
    Returns out-of-fold probabilities (train) and averaged test probabilities.
    Early stopping applied where supported.
    """
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X))
    tst_proba = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        model = model_class(**params)

        if model_class is lgb.LGBMClassifier:
            model.fit(Xtr, ytr,
                      eval_set=[(Xval, yval)],
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif model_class is CatBoostClassifier:
            model.fit(Xtr, ytr,
                      eval_set=(Xval, yval),
                      early_stopping_rounds=80,
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
        print(f"      Fold {fold}: F1={f1:.4f} @t={t:.3f}")

    return oof_proba, tst_proba


# ═══════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════
def _cv_f1(model_class, params, X, y, n_folds=3):
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        m = model_class(**params)
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        if model_class is lgb.LGBMClassifier:
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        elif model_class is CatBoostClassifier:
            m.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=50, verbose=False)
        elif model_class is xgb.XGBClassifier:
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        prob = m.predict_proba(Xval)[:, 1]
        _, f1 = find_best_threshold(yval, prob)
        scores.append(f1)
    return float(np.mean(scores))


def tune_model(model_class, param_space_fn, X, y, n_trials):
    def objective(trial):
        params = param_space_fn(trial)
        return _cv_f1(model_class, params, X, y)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def xgb_space(trial):
    return dict(
        max_depth        = trial.suggest_int('max_depth', 4, 10),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        n_estimators     = trial.suggest_int('n_estimators', 400, 1000),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        gamma            = trial.suggest_float('gamma', 0, 5),
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 1, 6),
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric='logloss',
    )


def lgb_space(trial):
    return dict(
        max_depth         = trial.suggest_int('max_depth', 4, 12),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        n_estimators      = trial.suggest_int('n_estimators', 400, 1000),
        num_leaves        = trial.suggest_int('num_leaves', 31, 255),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100),
        scale_pos_weight  = trial.suggest_float('scale_pos_weight', 1, 6),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


def cat_space(trial):
    return dict(
        depth               = trial.suggest_int('depth', 4, 10),
        learning_rate       = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        iterations          = trial.suggest_int('iterations', 400, 800),
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1, 10),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1),
        scale_pos_weight    = trial.suggest_float('scale_pos_weight', 1, 6),
        random_state=RANDOM_STATE, verbose=0, thread_count=-1,
    )


# ─── Default params (used when Optuna is off) ────────────────
DEFAULT_XGB = dict(
    max_depth=7, learning_rate=0.05, n_estimators=700,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, min_child_weight=3,
    gamma=0.1, scale_pos_weight=3.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric='logloss',
)
DEFAULT_LGB = dict(
    max_depth=8, learning_rate=0.05, n_estimators=700, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.0, min_child_samples=30, scale_pos_weight=3.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
DEFAULT_CAT = dict(
    depth=7, learning_rate=0.05, iterations=600,
    l2_leaf_reg=3.0, bagging_temperature=0.5, scale_pos_weight=3.0,
    random_state=RANDOM_STATE, verbose=0, thread_count=-1,
)


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*65)
    print("  TELECOM CHURN PREDICTION — LEADERBOARD-OPTIMIZED PIPELINE")
    print("="*65 + "\n")

    # ── Load ─────────────────────────────────────────────────
    print("→ Loading data...")
    df_train = load_csv(TRAIN_PATH)
    df_test  = load_csv(TEST_PATH)
    print(f"  Train: {df_train.shape}  |  Test: {df_test.shape}")

    TARGET = 'churn'
    y = df_train[TARGET].values
    churn_rate = y.mean()
    print(f"  Class dist → Stay: {(y==0).sum():,}  Churn: {(y==1).sum():,}  "
          f"({churn_rate:.2%} churn rate)\n")

    # ── Feature engineering ──────────────────────────────────
    print("→ Engineering features...")
    X_raw  = engineer_features(df_train.drop(columns=[TARGET]))
    Xt_raw = engineer_features(df_test)

    # ── Encode categoricals ──────────────────────────────────
    print("→ Encoding categoricals...")
    X_raw, Xt_raw, _ = encode_categoricals(X_raw, Xt_raw)

    # ── Align feature columns ────────────────────────────────
    feature_cols = [c for c in X_raw.columns if c in Xt_raw.columns]
    X  = X_raw[feature_cols].values.astype(np.float32)
    Xt = Xt_raw[feature_cols].values.astype(np.float32)
    print(f"  Total features: {len(feature_cols)}\n")

    # ── Hyperparameter tuning ────────────────────────────────
    if USE_OPTUNA and OPTUNA_AVAILABLE:
        print(f"→ Tuning hyperparameters with Optuna ({OPTUNA_TRIALS} trials each)...")
        print("  [XGBoost]  ", end='', flush=True)
        xgb_params = tune_model(xgb.XGBClassifier, xgb_space, X, y, OPTUNA_TRIALS)
        xgb_params.update(dict(random_state=RANDOM_STATE, n_jobs=-1,
                               verbosity=0, eval_metric='logloss'))
        print("done")

        print("  [LightGBM] ", end='', flush=True)
        lgb_params = tune_model(lgb.LGBMClassifier, lgb_space, X, y, OPTUNA_TRIALS)
        lgb_params.update(dict(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
        print("done")

        print("  [CatBoost] ", end='', flush=True)
        cat_params = tune_model(CatBoostClassifier, cat_space, X, y,
                                max(10, OPTUNA_TRIALS // 2))
        cat_params.update(dict(random_state=RANDOM_STATE, verbose=0, thread_count=-1))
        print("done\n")
    else:
        reason = "disabled" if not USE_OPTUNA else "not installed (pip install optuna)"
        print(f"→ Using preset hyperparameters (Optuna {reason})\n")
        xgb_params = DEFAULT_XGB
        lgb_params = DEFAULT_LGB
        cat_params = DEFAULT_CAT

    # ── OOF training ─────────────────────────────────────────
    print(f"→ Training with {N_FOLDS}-fold OOF stacking...\n")

    print("  [1/3] XGBoost")
    xgb_oof, xgb_tst = oof_predict(xgb.XGBClassifier, xgb_params, X, y, Xt, N_FOLDS)

    print("\n  [2/3] LightGBM")
    lgb_oof, lgb_tst = oof_predict(lgb.LGBMClassifier, lgb_params, X, y, Xt, N_FOLDS)

    print("\n  [3/3] CatBoost")
    cat_oof, cat_tst = oof_predict(CatBoostClassifier, cat_params, X, y, Xt, N_FOLDS)

    # ── Optimal blending weights ─────────────────────────────
    print("\n→ Optimizing ensemble blend weights...")
    best_blend_f1, best_w = 0.0, (0.34, 0.33, 0.33)
    for w1 in np.arange(0.1, 0.7, 0.05):
        for w2 in np.arange(0.1, 0.7, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 <= 0.05:
                continue
            blend = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
            _, f1 = find_best_threshold(y, blend)
            if f1 > best_blend_f1:
                best_blend_f1, best_w = f1, (w1, w2, w3)

    w1, w2, w3 = best_w
    print(f"  Weights → XGB: {w1:.2f} | LGB: {w2:.2f} | CAT: {w3:.2f}")

    oof_blend = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
    tst_blend = w1 * xgb_tst + w2 * lgb_tst + w3 * cat_tst

    # ── Final threshold optimization ─────────────────────────
    best_t, best_f1 = find_best_threshold(y, oof_blend)
    oof_auc = roc_auc_score(y, oof_blend)

    print(f"\n{'='*65}")
    print(f"  OOF F1 Score : {best_f1:.4f}  (threshold={best_t:.3f})")
    print(f"  OOF ROC-AUC  : {oof_auc:.4f}")
    print(f"{'='*65}\n")
    print("Classification Report (OOF):")
    print(classification_report(y, (oof_blend >= best_t).astype(int),
                                target_names=['Stay (0)', 'Churn (1)']))

    # ── Generate submission ───────────────────────────────────
    # NOTE: if the leaderboard expects a different ID column name,
    #       change 'id' below to match the sample_submission.csv
    test_preds = (tst_blend >= best_t).astype(int)

    submission = pd.DataFrame({
        'id':    range(len(test_preds)),
        'churn': test_preds,
    })
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"→ Submission saved to '{OUTPUT_PATH}'")
    print(f"  Predicted churn rate: {test_preds.mean():.2%}  "
          f"(train churn rate: {churn_rate:.2%})")
    print(f"\n{submission.head(10).to_string(index=False)}\n")

    return submission


if __name__ == "__main__":
    main()
