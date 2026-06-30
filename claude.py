"""
=============================================================================
Telecom Churn Prediction — CatBoost Pipeline
=============================================================================
Goal  : Maximise F1-score on the minority (churn=1) class.
Author: Senior Data Scientist template
Deps  : catboost, optuna, imbalanced-learn, scikit-learn, pandas, numpy,
        matplotlib, seaborn
Install:
    pip install catboost optuna imbalanced-learn scikit-learn \
                pandas numpy matplotlib seaborn
=============================================================================
"""

# ── Stdlib ──────────────────────────────────────────────────────────────────
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import StratifiedKFold
from sklearn.metrics          import (f1_score, classification_report,
                                      confusion_matrix)
from sklearn.experimental     import enable_iterative_imputer   # noqa: F401
from sklearn.impute           import IterativeImputer
from sklearn.preprocessing    import LabelEncoder

from catboost import CatBoostClassifier, Pool

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optional — comment the block out if imblearn is not installed
try:
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    log.warning("imbalanced-learn not found — SMOTE-Tomek block will be skipped.")

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_PATH   = "training_dataset.csv"
TEST_PATH    = "testing_dataset.csv"
TARGET_COL   = "churn"
RANDOM_STATE = 42
N_SPLITS     = 5           # StratifiedKFold folds
N_TRIALS     = 30          # Optuna trials (raise to 60-100 for production)
USE_SMOTE    = False       # Set True to enable SMOTE-Tomek resampling
OUTPUT_DIR   = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(train_path: str, test_path: str):
    """Load raw CSV files and return DataFrames."""
    log.info("Loading data …")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    log.info(f"  Train : {train.shape}  |  Test : {test.shape}")
    return train, test


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def detect_categorical_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    Auto-detect categorical columns.
    Why: CatBoost handles cats natively — passing them avoids one-hot explosion
    and lets the model find better splits on high-cardinality features.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target]
    log.info(f"  Detected {len(cat_cols)} categorical columns: {cat_cols}")
    return cat_cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature engineering before imputation.
    Why: Derived features built on raw data before imputation avoid leakage
    and can capture signal that raw columns miss.
    """
    df = df.copy()

    # Tenure in months — smoother than raw days for tree splits
    if "customer_tenure_days" in df.columns:
        df["tenure_months"] = df["customer_tenure_days"] / 30.44

    # Registration year / month — captures cohort effects on churn
    if "date_of_registration" in df.columns:
        reg = pd.to_datetime(df["date_of_registration"], errors="coerce")
        df["reg_year"]  = reg.dt.year.astype("Int64")
        df["reg_month"] = reg.dt.month.astype("Int64")
        df.drop(columns=["date_of_registration"], inplace=True)

    return df


def impute_missing(train: pd.DataFrame,
                   test:  pd.DataFrame,
                   num_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use IterativeImputer (MICE) on numeric columns.

    Why IterativeImputer > simple mean:
    - Models each missing column as a function of all other columns.
    - Preserves correlations between features (e.g. sms_sent ~ data_used).
    - Reduces bias introduced by single-value imputation, which directly
      hurts F1 by feeding the model distorted signal.
    """
    log.info("  Imputing missing values with IterativeImputer (MICE) …")

    # ── Guard: replace ±inf with NaN so IterativeImputer can handle them ────
    # inf values arise from ratio features (e.g. division by zero in
    # salary_data_ratio). IterativeImputer raises ValueError on inf, so we
    # convert them to NaN first — they will then be imputed like any other
    # missing value.
    for col in num_cols:
        if col in train.columns:
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        if col in test.columns:
            test[col]  = test[col].replace([np.inf, -np.inf], np.nan)

    # ── Guard: clip extreme outliers that can destabilise MICE ──────────────
    # Values beyond 1e15 in float64 are effectively meaningless for a telecom
    # dataset and cause numerical overflow inside the imputer's internal
    # BayesianRidge estimator.
    for col in num_cols:
        if col in train.columns:
            cap = train[col].quantile(0.9999)
            floor = train[col].quantile(0.0001)
            train[col] = train[col].clip(lower=floor, upper=cap)
        if col in test.columns:
            test[col]  = test[col].clip(lower=floor, upper=cap)

    n_inf_fixed = train[num_cols].isnull().sum().sum()
    log.info(f"  Total NaN (after inf→NaN conversion) to impute: {n_inf_fixed:,}")

    imp = IterativeImputer(
        max_iter=10,
        random_state=RANDOM_STATE,
        initial_strategy="median",   # robust to outliers vs mean
    )

    # Fit on train only — prevents test-set leakage
    train[num_cols] = imp.fit_transform(train[num_cols])
    test[num_cols]  = imp.transform(test[num_cols])

    return train, test


def encode_categoricals_for_smote(df: pd.DataFrame,
                                   cat_cols: list[str]) -> pd.DataFrame:
    """
    Label-encode categoricals only when SMOTE is used.
    SMOTE requires purely numeric input; CatBoost's Pool handles raw strings.
    """
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def preprocess(train_raw: pd.DataFrame,
               test_raw:  pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline.
    Returns a dict with everything the modelling steps need.
    """
    log.info("Preprocessing …")

    train = engineer_features(train_raw)
    test  = engineer_features(test_raw)

    cat_cols = detect_categorical_columns(train, TARGET_COL)
    num_cols = [c for c in train.select_dtypes(include=np.number).columns
                if c != TARGET_COL]

    train, test = impute_missing(train, test, num_cols)

    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    # CatBoost needs column indices, not names, when using Pool
    cat_feat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    # Fill remaining NaNs in cat cols with "Unknown"
    for col in cat_cols:
        if col in X.columns:
            X[col]    = X[col].fillna("Unknown").astype(str)
            test[col] = test[col].fillna("Unknown").astype(str)

    return {
        "X": X, "y": y,
        "X_test": test,
        "cat_cols": cat_cols,
        "cat_feat_indices": cat_feat_indices,
    }


# =============================================================================
# 3. CLASS IMBALANCE
# =============================================================================

def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    scale_pos_weight = negatives / positives.

    Why: This tells CatBoost to penalise mis-classifying the minority class
    more heavily, shifting precision-recall balance toward higher recall on
    churn=1, which lifts the F1 score on the minority class.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos
    log.info(f"  Class counts  →  neg={neg:,}  pos={pos:,}  "
             f"scale_pos_weight={spw:.3f}")
    return spw


def apply_smote_tomek(X: pd.DataFrame, y: pd.Series,
                       cat_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Optional SMOTE-Tomek resampling.

    Why SMOTE-Tomek > plain SMOTE:
    - SMOTE synthesises minority samples to over-sample.
    - Tomek links remove borderline majority samples near the boundary.
    - Together they create a cleaner decision boundary, boosting F1.

    Note: Apply ONLY on the training fold, never on validation/test data —
    doing so would leak distribution information and inflate metrics.
    """
    if not SMOTE_AVAILABLE:
        log.warning("SMOTE-Tomek skipped — imbalanced-learn not installed.")
        return X, y

    log.info("  Applying SMOTE-Tomek …")
    X_enc = encode_categoricals_for_smote(X, cat_cols)
    smt = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smt.fit_resample(X_enc, y)
    log.info(f"  After SMOTE-Tomek: {pd.Series(y_res).value_counts().to_dict()}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)


# =============================================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series,
                          cat_feat_indices: list[int],
                          scale_pos_weight: float) -> dict:
    """
    Use Optuna (TPE sampler) to maximise mean CV F1 on the minority class.

    Why tune?
    - learning_rate × iterations control under/over-fit trade-off.
    - depth controls tree expressiveness — too deep = overfit on imbalanced data.
    - l2_leaf_reg (L2 regularisation) reduces variance, helping generalisation.
    - border_count affects numeric feature quantisation granularity.
    """
    log.info(f"Tuning hyperparameters with Optuna ({N_TRIALS} trials) …")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth"         : trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg"   : trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "iterations"    : trial.suggest_int("iterations", 200, 1000, step=100),
            "border_count"  : trial.suggest_categorical("border_count", [32, 64, 128]),
            # Fixed params
            "loss_function"    : "Logloss",
            "eval_metric"      : "F1",
            "scale_pos_weight" : scale_pos_weight,
            "random_seed"      : RANDOM_STATE,
            "verbose"          : False,
            "allow_writing_files": False,
        }

        fold_f1s = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if USE_SMOTE:
                X_tr, y_tr = apply_smote_tomek(X_tr, y_tr, [
                    X.columns[i] for i in cat_feat_indices])

            pool_tr  = Pool(X_tr,  y_tr,  cat_features=cat_feat_indices)
            pool_val = Pool(X_val, y_val, cat_features=cat_feat_indices)

            model = CatBoostClassifier(**params)
            model.fit(pool_tr, eval_set=pool_val,
                      early_stopping_rounds=50, verbose=False)

            proba = model.predict_proba(X_val)[:, 1]
            # Quick threshold search within the trial
            best = max(
                (f1_score(y_val, (proba >= t).astype(int), zero_division=0), t)
                for t in np.arange(0.1, 0.91, 0.05)
            )
            fold_f1s.append(best[0])

        return float(np.mean(fold_f1s))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    log.info(f"  Best params : {best}")
    log.info(f"  Best CV F1  : {study.best_value:.4f}")
    return best


# =============================================================================
# 5. OPTIMAL THRESHOLD SEARCH
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray,
                            y_proba: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds [0.10 → 0.90] and pick the one maximising F1.

    Why this matters:
    - CatBoost (and any classifier) outputs P(churn=1), not hard labels.
    - Default threshold = 0.5, which is calibrated for balanced data.
    - On imbalanced data the optimal F1 threshold is almost always < 0.5.
    - Moving the threshold down increases recall on the minority class,
      often more than the precision loss — net F1 improves significantly.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
              for t in thresholds]
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]


# =============================================================================
# 6. FINAL TRAINING & EVALUATION
# =============================================================================

def train_final_model(X: pd.DataFrame, y: pd.Series,
                       best_params: dict,
                       cat_feat_indices: list[int],
                       scale_pos_weight: float) -> dict:
    """
    Re-train with best params using StratifiedKFold, collect OOF predictions.

    Why OOF (Out-Of-Fold) predictions:
    - Gives an unbiased probability estimate for every training sample
      without data leakage.
    - We use these OOF probas to find the optimal threshold on all data,
      giving a stable estimate before applying to the test set.
    """
    log.info("Training final model with StratifiedKFold …")

    final_params = {
        **best_params,
        "loss_function"      : "Logloss",
        "eval_metric"        : "F1",
        "scale_pos_weight"   : scale_pos_weight,
        "random_seed"        : RANDOM_STATE,
        "verbose"            : False,
        "allow_writing_files": False,
    }

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE)
    oof_proba  = np.zeros(len(y))
    fold_f1s   = []
    models     = []
    feat_imps  = np.zeros(X.shape[1])

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if USE_SMOTE:
            cat_names = [X.columns[i] for i in cat_feat_indices]
            X_tr, y_tr = apply_smote_tomek(X_tr, y_tr, cat_names)

        pool_tr  = Pool(X_tr,  y_tr,  cat_features=cat_feat_indices)
        pool_val = Pool(X_val, y_val, cat_features=cat_feat_indices)

        model = CatBoostClassifier(**final_params)
        model.fit(pool_tr, eval_set=pool_val,
                  early_stopping_rounds=50, verbose=False)

        proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba

        threshold, f1 = find_optimal_threshold(y_val.values, proba)
        fold_f1s.append(f1)
        log.info(f"  Fold {fold}: threshold={threshold:.2f}  F1={f1:.4f}")

        models.append(model)
        feat_imps += model.get_feature_importance()

    feat_imps /= N_SPLITS

    # ── Global threshold on OOF ─────────────────────────────────────────────
    opt_threshold, oof_f1 = find_optimal_threshold(y.values, oof_proba)
    log.info(f"\n  OOF F1 @ threshold={opt_threshold:.2f} : {oof_f1:.4f}")
    log.info(f"  Mean fold F1 : {np.mean(fold_f1s):.4f} "
             f"± {np.std(fold_f1s):.4f}")

    return {
        "models"        : models,
        "oof_proba"     : oof_proba,
        "opt_threshold" : opt_threshold,
        "oof_f1"        : oof_f1,
        "fold_f1s"      : fold_f1s,
        "feat_imps"     : feat_imps,
        "feat_names"    : X.columns.tolist(),
    }


# =============================================================================
# 7. EVALUATION PLOTS
# =============================================================================

def evaluate_and_plot(y_true: np.ndarray,
                       oof_proba: np.ndarray,
                       opt_threshold: float,
                       feat_imps: np.ndarray,
                       feat_names: list[str]) -> None:
    """
    Produce:
    1. Classification report (printed)
    2. Confusion matrix heatmap
    3. Feature importance bar chart
    4. Threshold vs F1 curve
    """
    y_pred = (oof_proba >= opt_threshold).astype(int)

    # ── 1. Classification report ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("CLASSIFICATION REPORT  (OOF predictions)")
    log.info("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=["No Churn", "Churn"]))

    # ── 2. Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    axes[0].set_title("Confusion Matrix (OOF)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # ── 3. Feature importance ───────────────────────────────────────────────
    fi_df = (pd.DataFrame({"feature": feat_names, "importance": feat_imps})
               .sort_values("importance", ascending=True)
               .tail(20))

    axes[1].barh(fi_df["feature"], fi_df["importance"], color="steelblue")
    axes[1].set_title("Top-20 Feature Importances (avg over folds)")
    axes[1].set_xlabel("Importance")

    # ── 4. Threshold curve ──────────────────────────────────────────────────
    thresholds = np.arange(0.10, 0.91, 0.01)
    f1_scores  = [f1_score(y_true, (oof_proba >= t).astype(int),
                           zero_division=0) for t in thresholds]

    axes[2].plot(thresholds, f1_scores, color="darkorange", linewidth=2)
    axes[2].axvline(opt_threshold, color="red", linestyle="--",
                    label=f"Optimal = {opt_threshold:.2f}")
    axes[2].set_title("F1 Score vs Decision Threshold")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "evaluation_plots.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Plots saved → {out_path}")


# =============================================================================
# 8. TEST SET PREDICTIONS
# =============================================================================

def predict_test_set(models: list,
                      X_test: pd.DataFrame,
                      cat_feat_indices: list[int],
                      opt_threshold: float) -> pd.DataFrame:
    """
    Average probabilities across all fold models (ensemble).

    Why ensemble:
    - Each fold model saw a different 80% of training data.
    - Averaging their probabilities reduces variance and typically
      outperforms any single model by 1-2 F1 points.
    """
    log.info("Generating test set predictions …")
    probas = np.mean(
        [m.predict_proba(X_test)[:, 1] for m in models], axis=0
    )
    preds = (probas >= opt_threshold).astype(int)

    result = pd.DataFrame({
        "predicted_proba": probas,
        "predicted_churn": preds,
    })
    out_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    result.to_csv(out_path, index=False)
    log.info(f"  Saved {len(result):,} predictions → {out_path}")
    log.info(f"  Predicted churn rate: {preds.mean():.3%}")
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("TELECOM CHURN — CatBoost Pipeline")
    log.info("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)

    # ── Preprocess ───────────────────────────────────────────────────────────
    pp = preprocess(train_raw, test_raw)
    X, y           = pp["X"], pp["y"]
    X_test         = pp["X_test"]
    cat_feat_idx   = pp["cat_feat_indices"]

    # ── Class imbalance weight ───────────────────────────────────────────────
    spw = compute_scale_pos_weight(y)

    # ── Hyperparameter tuning ────────────────────────────────────────────────
    best_params = tune_hyperparameters(X, y, cat_feat_idx, spw)

    # ── Final training ───────────────────────────────────────────────────────
    results = train_final_model(X, y, best_params, cat_feat_idx, spw)

    # ── Evaluation ───────────────────────────────────────────────────────────
    evaluate_and_plot(
        y_true       = y.values,
        oof_proba    = results["oof_proba"],
        opt_threshold= results["opt_threshold"],
        feat_imps    = results["feat_imps"],
        feat_names   = results["feat_names"],
    )

    # ── Test predictions ─────────────────────────────────────────────────────
    predict_test_set(
        models          = results["models"],
        X_test          = X_test,
        cat_feat_indices= cat_feat_idx,
        opt_threshold   = results["opt_threshold"],
    )

    log.info("\n✓ Pipeline complete.")
    log.info(f"  Final OOF F1 : {results['oof_f1']:.4f}  "
             f"(threshold = {results['opt_threshold']:.2f})")


if __name__ == "__main__":
    main()