"""
Churn Prediction Model - Kaggle Competition
=============================================
Ensemble pipeline with robust feature engineering, target encoding,
and gradient boosting models (CatBoost, XGBoost, LightGBM).

F1-OPTIMIZED VERSION:
- CatBoost / LightGBM early stopping driven by F1 instead of AUC
- Ensemble weights AND threshold chosen jointly to maximize F1
  (instead of weighting by AUC and threshold-sweeping after the fact)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_recall_curve

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 60)
print("1. Loading Data...")
print("=" * 60)

train = pd.read_csv('training_dataset.csv')
test = pd.read_csv('testing_dataset.csv')

print(f"Training set: {train.shape}")
print(f"Testing set:  {test.shape}")
print(f"\nTarget distribution:\n{train['churn'].value_counts(normalize=True)}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 60)
print("2. Feature Engineering...")
print("=" * 60)

TARGET = 'churn'


def safe_divide(a, b, fill=0):
    """Safe division that replaces inf/nan with fill value."""
    result = a / b
    result = result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return result


def engineer_features(df, is_train=True):
    """Create features from existing data with robust inf/nan handling."""
    df = df.copy()

    # --- Handle missing values first ---
    df['estimated_salary'] = df['estimated_salary'].fillna(df['estimated_salary'].median())
    df['sms_sent'] = df['sms_sent'].fillna(df['sms_sent'].median())
    df['data_used'] = df['data_used'].fillna(df['data_used'].median())

    # --- Fix the existing salary_data_ratio (has inf) ---
    df['salary_data_ratio'] = safe_divide(df['estimated_salary'], df['data_used'])

    # --- Date features ---
    df['date_of_registration'] = pd.to_datetime(df['date_of_registration'])
    df['reg_year'] = df['date_of_registration'].dt.year
    df['reg_month'] = df['date_of_registration'].dt.month
    df['reg_day_of_week'] = df['date_of_registration'].dt.dayofweek
    df['reg_quarter'] = df['date_of_registration'].dt.quarter
    df['reg_is_weekend'] = (df['date_of_registration'].dt.dayofweek >= 5).astype(int)
    df['reg_day_of_year'] = df['date_of_registration'].dt.dayofyear
    df['reg_week_of_year'] = df['date_of_registration'].dt.isocalendar().week.astype(int)

    # Days since registration relative to a reference
    reference_date = pd.Timestamp('2026-06-17')
    df['days_since_registration'] = (reference_date - df['date_of_registration']).dt.days
    df.drop('date_of_registration', axis=1, inplace=True)

    # --- Communication features ---
    df['total_comm'] = df['calls_made'] + df['sms_sent']
    df['calls_sms_ratio'] = safe_divide(df['calls_made'], df['sms_sent'] + 1)
    df['sms_calls_ratio'] = safe_divide(df['sms_sent'], df['calls_made'] + 1)
    df['comm_per_day'] = safe_divide(df['total_comm'], df['customer_tenure_days'] + 1)
    df['calls_pct'] = safe_divide(df['calls_made'], df['total_comm'] + 1)
    df['sms_pct'] = safe_divide(df['sms_sent'], df['total_comm'] + 1)

    # --- Data usage features ---
    df['data_per_tenure'] = safe_divide(df['data_used'], df['customer_tenure_days'] + 1)
    df['data_calls_ratio'] = safe_divide(df['data_used'], df['calls_made'] + 1)

    # --- Financial features ---
    df['salary_per_dependent'] = safe_divide(df['estimated_salary'], df['num_dependents'] + 1)
    df['salary_comm_ratio'] = safe_divide(df['estimated_salary'], df['total_comm'] + 1)
    df['salary_per_tenure'] = safe_divide(df['estimated_salary'], df['customer_tenure_days'] + 1)

    # --- Activity features ---
    df['activity_per_tenure'] = safe_divide(df['activity_score'], df['customer_tenure_days'] + 1)
    df['activity_per_call'] = safe_divide(df['activity_score'], df['calls_made'] + 1)
    df['activity_salary_ratio'] = safe_divide(df['activity_score'], df['estimated_salary'] + 1)
    df['activity_data_ratio'] = safe_divide(df['activity_score'], df['data_used'] + 1)

    # --- Demographic flags ---
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_young'] = (df['age'] <= 25).astype(int)
    df['is_middle_aged'] = ((df['age'] > 35) & (df['age'] <= 55)).astype(int)

    # --- Tenure flags ---
    df['tenure_years'] = df['customer_tenure_days'] / 365.25
    df['is_new_customer'] = (df['customer_tenure_days'] <= 365).astype(int)
    df['is_long_term'] = (df['customer_tenure_days'] >= 1460).astype(int)

    # --- Polynomial interactions ---
    df['age_x_tenure'] = df['age'] * df['customer_tenure_days']
    df['salary_x_data'] = df['estimated_salary'] * df['data_used']
    df['calls_x_sms'] = df['calls_made'] * df['sms_sent']
    df['deps_x_salary'] = df['num_dependents'] * df['estimated_salary']
    df['age_x_salary'] = df['age'] * df['estimated_salary']
    df['age_x_activity'] = df['age'] * df['activity_score']
    df['tenure_x_activity'] = df['customer_tenure_days'] * df['activity_score']

    # --- Log transforms ---
    for col in ['estimated_salary', 'data_used', 'activity_score',
                'customer_tenure_days', 'calls_made', 'sms_sent']:
        # Shift to handle negatives before log
        min_val = df[col].min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        df[f'{col}_log'] = np.log1p(df[col] + shift)

    # --- Squared features ---
    df['age_sq'] = df['age'] ** 2
    df['tenure_sq'] = df['customer_tenure_days'] ** 2
    df['salary_sq'] = df['estimated_salary'] ** 2

    # --- Binned features ---
    df['age_bin'] = pd.cut(df['age'], bins=10, labels=False)
    df['salary_bin'] = pd.cut(df['estimated_salary'], bins=10, labels=False)
    df['tenure_bin'] = pd.cut(df['customer_tenure_days'], bins=10, labels=False)
    df['data_bin'] = pd.cut(df['data_used'], bins=10, labels=False)

    # Final cleanup: replace any remaining inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# Apply feature engineering
train = engineer_features(train, is_train=True)
test = engineer_features(test, is_train=False)

print(f"Engineered training set: {train.shape}")
print(f"Engineered testing set:  {test.shape}")

# ============================================================================
# 3. ENCODE CATEGORICALS + TARGET ENCODING
# ============================================================================
print("\n" + "=" * 60)
print("3. Encoding Categorical Features...")
print("=" * 60)

categorical_cols = ['telecom_partner', 'gender', 'state', 'city']

# --- Label Encoding ---
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col + '_le'] = le.transform(train[col].astype(str))
    test[col + '_le'] = le.transform(test[col].astype(str))
    label_encoders[col] = le
    print(f"  Label encoded '{col}': {len(le.classes_)} unique values")

# --- Frequency Encoding ---
for col in categorical_cols:
    freq = train[col].value_counts(normalize=True).to_dict()
    train[col + '_freq'] = train[col].map(freq)
    test[col + '_freq'] = test[col].map(freq).fillna(0)

# --- Target Encoding (with regularization via K-fold to avoid leakage) ---
print("\n  Computing target encodings (5-fold regularized)...")
N_FOLDS_TE = 5
skf_te = StratifiedKFold(n_splits=N_FOLDS_TE, shuffle=True, random_state=99)
global_mean = train[TARGET].mean()

for col in categorical_cols:
    train[col + '_te'] = global_mean
    for fold, (tr_idx, val_idx) in enumerate(skf_te.split(train, train[TARGET])):
        means = train.iloc[tr_idx].groupby(col)[TARGET].mean()
        # Regularized: blend with global mean based on count
        counts = train.iloc[tr_idx].groupby(col)[TARGET].count()
        smooth = 20  # smoothing factor
        smoothed_means = (means * counts + global_mean * smooth) / (counts + smooth)
        train.loc[train.index[val_idx], col + '_te'] = train.iloc[val_idx][col].map(smoothed_means)

    # For test set, use full training data
    means = train.groupby(col)[TARGET].mean()
    counts = train.groupby(col)[TARGET].count()
    smoothed_means = (means * counts + global_mean * smooth) / (counts + smooth)
    test[col + '_te'] = test[col].map(smoothed_means).fillna(global_mean)
    print(f"  Target encoded '{col}'")

# --- Combination Target Encodings ---
combo_cols = [('telecom_partner', 'city'), ('state', 'city'), ('telecom_partner', 'state')]
for col1, col2 in combo_cols:
    combo_name = f'{col1}_{col2}'
    train[combo_name] = train[col1].astype(str) + '_' + train[col2].astype(str)
    test[combo_name] = test[col1].astype(str) + '_' + test[col2].astype(str)

    train[combo_name + '_te'] = global_mean
    for fold, (tr_idx, val_idx) in enumerate(skf_te.split(train, train[TARGET])):
        means = train.iloc[tr_idx].groupby(combo_name)[TARGET].mean()
        counts = train.iloc[tr_idx].groupby(combo_name)[TARGET].count()
        smoothed_means = (means * counts + global_mean * smooth) / (counts + smooth)
        train.loc[train.index[val_idx], combo_name + '_te'] = train.iloc[val_idx][combo_name].map(smoothed_means)

    means = train.groupby(combo_name)[TARGET].mean()
    counts = train.groupby(combo_name)[TARGET].count()
    smoothed_means = (means * counts + global_mean * smooth) / (counts + smooth)
    test[combo_name + '_te'] = test[combo_name].map(smoothed_means).fillna(global_mean)
    print(f"  Target encoded combo '{combo_name}'")

    # Drop the string combo column
    train.drop(combo_name, axis=1, inplace=True)
    test.drop(combo_name, axis=1, inplace=True)

# Drop original categorical string columns (keep encoded versions)
train.drop(categorical_cols, axis=1, inplace=True)
test.drop(categorical_cols, axis=1, inplace=True)

# ============================================================================
# 4. PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "=" * 60)
print("4. Preparing Data...")
print("=" * 60)

EXCLUDE_COLS = [TARGET]
feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()

# Final safety check for inf and NaN
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Features: {len(feature_cols)}")
print(f"Training samples: {X.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Target ratio: {y.mean():.4f}")
print(f"Any inf in X: {np.isinf(X.values).any()}")
print(f"Any NaN in X: {np.isnan(X.values).any()}")

# ============================================================================
# 5. MODEL TRAINING WITH STRATIFIED K-FOLD
# ============================================================================
print("\n" + "=" * 60)
print("5. Training Ensemble Models with 5-Fold CV...")
print("=" * 60)

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = {}
test_preds = {}

# -------------------------
# 5a. CatBoost  (early stopping now driven by F1, not AUC)
# -------------------------
from catboost import CatBoostClassifier

print("\n--- CatBoost ---")
cb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))

# Identify label-encoded categorical column indices
cat_feature_names = [c + '_le' for c in ['telecom_partner', 'gender', 'state', 'city']]
cat_indices = [list(X.columns).index(c) for c in cat_feature_names if c in X.columns]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold + 1}/{N_FOLDS}...", end=" ", flush=True)
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        min_data_in_leaf=50,
        subsample=0.7,
        colsample_bylevel=0.7,
        random_strength=2,
        bagging_temperature=1.0,
        eval_metric='F1',          # <-- changed from 'AUC': stop training where F1 peaks
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=300,
        cat_features=cat_indices,
        auto_class_weights='Balanced',
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)

    cb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    cb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, cb_oof[val_idx])
    fold_f1 = f1_score(y_val, (cb_oof[val_idx] >= 0.5).astype(int))
    print(f"AUC = {fold_auc:.6f}  F1@0.5 = {fold_f1:.6f} (best iter: {model.best_iteration_})")

cb_auc = roc_auc_score(y, cb_oof)
print(f"  CatBoost Overall OOF AUC: {cb_auc:.6f}")
oof_preds['catboost'] = cb_oof
test_preds['catboost'] = cb_test

# -------------------------
# 5b. LightGBM  (early stopping now driven by a custom F1 metric, not AUC)
# -------------------------
try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    def lgb_f1_eval(y_true, y_pred):
        """Custom eval metric for LightGBM sklearn API: (name, value, is_higher_better)."""
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return 'f1', f1_score(y_true, y_pred_binary), True

    print("\n--- LightGBM ---")
    lgbm_oof = np.zeros(len(X))
    lgbm_test = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{N_FOLDS}...", end=" ", flush=True)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=10,
            min_child_samples=50,
            is_unbalance=True,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=lgb_f1_eval,        # <-- custom F1 metric drives early stopping
            callbacks=[
                lgb.early_stopping(300, verbose=False),
                lgb.log_evaluation(0),
            ]
        )

        lgbm_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        lgbm_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_val, lgbm_oof[val_idx])
        fold_f1 = f1_score(y_val, (lgbm_oof[val_idx] >= 0.5).astype(int))
        print(f"AUC = {fold_auc:.6f}  F1@0.5 = {fold_f1:.6f} (best iter: {model.best_iteration_})")

    lgbm_auc = roc_auc_score(y, lgbm_oof)
    print(f"  LightGBM Overall OOF AUC: {lgbm_auc:.6f}")
    oof_preds['lightgbm'] = lgbm_oof
    test_preds['lightgbm'] = lgbm_test

except ImportError:
    print("LightGBM not available, skipping.")

# -------------------------
# 5c. XGBoost  (left on AUC early-stopping for stability across xgboost versions;
#               its contribution is still re-weighted for F1 in the ensemble step below)
# -------------------------
try:
    from xgboost import XGBClassifier

    print("\n--- XGBoost ---")
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    # Compute scale_pos_weight for imbalanced classes
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos = neg_count / pos_count

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{N_FOLDS}...", end=" ", flush=True)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=10,
            min_child_weight=50,
            gamma=0.5,
            scale_pos_weight=scale_pos,
            eval_metric='auc',
            random_state=42 + fold,
            verbosity=0,
            early_stopping_rounds=300,
            tree_method='hist',
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=0
        )

        xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        xgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
        fold_f1 = f1_score(y_val, (xgb_oof[val_idx] >= 0.5).astype(int))
        print(f"AUC = {fold_auc:.6f}  F1@0.5 = {fold_f1:.6f} (best iter: {model.best_iteration})")

    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"  XGBoost Overall OOF AUC: {xgb_auc:.6f}")
    oof_preds['xgboost'] = xgb_oof
    test_preds['xgboost'] = xgb_test

except ImportError:
    print("XGBoost not available, skipping.")

# ============================================================================
# 6. F1-OPTIMIZED ENSEMBLE WEIGHTS + THRESHOLD
# ============================================================================
print("\n" + "=" * 60)
print("6. Finding F1-Optimal Ensemble Weights & Threshold...")
print("=" * 60)


def best_f1_and_threshold(scores, y_true):
    """Exact best F1 + the threshold that achieves it, via the PR curve
    (no manual grid stepping, no missed optimum between grid points)."""
    precision, recall, thresholds_pr = precision_recall_curve(y_true, scores)
    denom = precision + recall
    f1_curve = np.divide(2 * precision * recall, denom,
                          out=np.zeros_like(denom), where=denom > 0)
    best_idx = int(np.argmax(f1_curve))
    # precision_recall_curve returns one fewer threshold than precision/recall
    # (the last point corresponds to threshold = +inf / recall = 0)
    best_thresh = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 1.0
    return f1_curve[best_idx], best_thresh


model_names = list(oof_preds.keys())
oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
test_matrix = np.column_stack([test_preds[name] for name in model_names])

# Diagnostics: how good is each individual model on F1 alone?
print("\n  Individual model AUC / best-possible F1:")
for i, name in enumerate(model_names):
    auc_i = roc_auc_score(y, oof_matrix[:, i])
    f1_i, thresh_i = best_f1_and_threshold(oof_matrix[:, i], y)
    print(f"    {name}: AUC = {auc_i:.6f}  |  best F1 = {f1_i:.6f} @ threshold {thresh_i:.3f}")

n_models = len(model_names)
step = 0.05
grid_vals = np.round(np.arange(0, 1 + 1e-9, step), 4)

if n_models == 1:
    candidates = [np.array([1.0])]
elif n_models == 2:
    candidates = [np.array([w, 1 - w]) for w in grid_vals]
elif n_models == 3:
    candidates = [np.array([w1, w2, 1 - w1 - w2])
                  for w1 in grid_vals for w2 in grid_vals
                  if w1 + w2 <= 1 + 1e-9]
else:
    # Random simplex search for >3 models (Dirichlet draws cover the space well)
    rng = np.random.default_rng(42)
    candidates = [row for row in rng.dirichlet(np.ones(n_models), size=4000)]

best_overall_f1 = -1.0
best_weights = None
best_threshold = 0.5

for w in candidates:
    if w.sum() <= 0:
        continue
    w = w / w.sum()
    blended = oof_matrix @ w
    f1_val, thresh_val = best_f1_and_threshold(blended, y)
    if f1_val > best_overall_f1:
        best_overall_f1 = f1_val
        best_weights = w
        best_threshold = thresh_val

weights = dict(zip(model_names, best_weights))
print(f"\n  F1-optimal ensemble weights: { {k: round(v, 4) for k, v in weights.items()} }")
print(f"  F1-optimal threshold:        {best_threshold:.4f}")
print(f"  Best ensemble OOF F1:        {best_overall_f1:.6f}")

ensemble_oof = oof_matrix @ best_weights
ensemble_test = test_matrix @ best_weights

ensemble_auc = roc_auc_score(y, ensemble_oof)
best_f1 = best_overall_f1
print(f"\n  Ensemble OOF AUC (for reference, not the optimization target): {ensemble_auc:.6f}")
print(f"\nClassification Report at threshold {best_threshold:.4f}:")
preds_binary = (ensemble_oof >= best_threshold).astype(int)
print(classification_report(y, preds_binary, target_names=['No Churn', 'Churn']))

# ============================================================================
# 7. GENERATE SUBMISSIONS
# ============================================================================
print("\n" + "=" * 60)
print("7. Generating Submissions...")
print("=" * 60)

# Binary submission with optimal threshold
test_binary = (ensemble_test >= best_threshold).astype(int)
submission = pd.DataFrame({
    'id': range(1, len(X_test) + 1),
    'churn': test_binary
})
submission.to_csv('submission.csv', index=False)

print(f"  submission.csv saved")
print(f"  Shape: {submission.shape}")
print(f"  Distribution:\n{submission['churn'].value_counts()}")
print(f"  Churn rate: {submission['churn'].mean():.4f}")

# Probability submission
submission_prob = pd.DataFrame({
    'id': range(1, len(X_test) + 1),
    'churn': ensemble_test
})
submission_prob.to_csv('submission_proba.csv', index=False)
print(f"\n  submission_proba.csv saved")

# Also save individual model submissions for experimentation
# (each uses its OWN F1-optimal threshold, not the ensemble's)
for i, name in enumerate(model_names):
    _, indiv_thresh = best_f1_and_threshold(oof_matrix[:, i], y)
    sub = pd.DataFrame({
        'id': range(1, len(X_test) + 1),
        'churn': (test_matrix[:, i] >= indiv_thresh).astype(int)
    })
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f"  submission_{name}.csv saved (threshold={indiv_thresh:.3f}, churn rate: {sub['churn'].mean():.4f})")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
