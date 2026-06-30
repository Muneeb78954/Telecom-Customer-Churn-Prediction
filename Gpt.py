# ============================================================
# TELECOM CHURN PREDICTION USING CATBOOST
# Optimized for F1 Score
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import matplotlib.pyplot as plt
import seaborn as sns

import optuna

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "training_dataset.csv"
TEST_PATH = "testing_dataset.csv"

TARGET_COLUMN = "Churn"   # <-- CHANGE IF NEEDED
RANDOM_STATE = 42
N_SPLITS = 5

USE_ITERATIVE_IMPUTER = False
USE_SMOTETOMEK = False

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape : {test_df.shape}")

# ============================================================
# AUTOMATIC CATEGORICAL DETECTION
# ============================================================

def detect_categorical_columns(df):
    """
    Detect categorical features automatically.

    CatBoost handles categorical variables natively,
    avoiding one-hot encoding and preserving information.
    """
    cat_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return cat_cols


# ============================================================
# DATA PREPARATION
# ============================================================

X = train_df.drop(columns=[TARGET_COLUMN])
y = train_df[TARGET_COLUMN]

X_test_final = test_df.copy()

categorical_cols = detect_categorical_columns(X)

print("\nCategorical Columns:")
print(categorical_cols)

# ============================================================
# OPTIONAL ITERATIVE IMPUTATION
# ============================================================

if USE_ITERATIVE_IMPUTER:

    print("\nApplying IterativeImputer...")

    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=10
    )

    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    test_numeric_cols = X_test_final[numeric_cols]
    X_test_final[numeric_cols] = imputer.transform(test_numeric_cols)

else:
    print("\nUsing CatBoost native missing value handling.")


# ============================================================
# CLASS IMBALANCE HANDLING
# ============================================================

negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

scale_pos_weight = negative_count / positive_count

print(f"\nScale Pos Weight = {scale_pos_weight:.4f}")

# ============================================================
# OPTIONAL SMOTE-TOMEK
# ============================================================

if USE_SMOTETOMEK:

    from imblearn.combine import SMOTETomek

    print("\nApplying SMOTE-Tomek...")

    X_temp = X.copy()

    # Encode categories temporarily
    for col in categorical_cols:
        X_temp[col] = X_temp[col].astype(str)

    X_temp = pd.get_dummies(X_temp)

    smt = SMOTETomek(random_state=RANDOM_STATE)

    X_resampled, y_resampled = smt.fit_resample(X_temp, y)

    print("After Resampling:")
    print(y_resampled.value_counts())

# ============================================================
# OPTIMAL THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Search thresholds from 0.1 to 0.9
    to maximize F1 score.

    This step frequently improves F1 by 3-10%.
    """

    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        preds = (probabilities >= threshold).astype(int)

        score = f1_score(y_true, preds)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    params = {

        "iterations": trial.suggest_int(
            "iterations",
            300,
            2000
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "depth": trial.suggest_int(
            "depth",
            4,
            10
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1,
            20
        ),

        "loss_function": "Logloss",

        "eval_metric": "F1",

        "scale_pos_weight": scale_pos_weight,

        "random_seed": RANDOM_STATE,

        "verbose": False
    }

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []

    for train_idx, valid_idx in skf.split(X, y):

        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]

        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        train_pool = Pool(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_cols
        )

        valid_pool = Pool(
            X_valid_fold,
            y_valid_fold,
            cat_features=categorical_cols
        )

        model = CatBoostClassifier(**params)

        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=100,
            verbose=False
        )

        probs = model.predict_proba(
            X_valid_fold
        )[:, 1]

        threshold, f1 = find_best_threshold(
            y_valid_fold,
            probs
        )

        fold_scores.append(f1)

    return np.mean(fold_scores)


# ============================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================

print("\nStarting Optuna Optimization...")

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

print("\nBest F1:")
print(study.best_value)

print("\nBest Parameters:")
print(study.best_params)

# ============================================================
# OUT-OF-FOLD TRAINING
# ============================================================

best_params = study.best_params

best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "scale_pos_weight": scale_pos_weight,
    "random_seed": RANDOM_STATE,
    "verbose": False
})

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_probs = np.zeros(len(X))

models = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

    print(f"\nFold {fold+1}")

    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]

    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]

    train_pool = Pool(
        X_train_fold,
        y_train_fold,
        cat_features=categorical_cols
    )

    valid_pool = Pool(
        X_valid_fold,
        y_valid_fold,
        cat_features=categorical_cols
    )

    model = CatBoostClassifier(**best_params)

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=100
    )

    probs = model.predict_proba(
        X_valid_fold
    )[:, 1]

    oof_probs[valid_idx] = probs

    models.append(model)

# ============================================================
# FIND GLOBAL BEST THRESHOLD
# ============================================================

best_threshold, best_f1 = find_best_threshold(
    y,
    oof_probs
)

print("\nOptimal Threshold:", round(best_threshold, 4))
print("OOF F1 Score:", round(best_f1, 4))

# ============================================================
# FINAL EVALUATION
# ============================================================

oof_preds = (
    oof_probs >= best_threshold
).astype(int)

print("\nClassification Report")
print(
    classification_report(
        y,
        oof_preds
    )
)

cm = confusion_matrix(
    y,
    oof_preds
)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# ============================================================
# TRAIN FINAL MODEL ON FULL DATA
# ============================================================

final_pool = Pool(
    X,
    y,
    cat_features=categorical_cols
)

final_model = CatBoostClassifier(
    **best_params
)

final_model.fit(final_pool)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.get_feature_importance()
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,8))

sns.barplot(
    data=feature_importance.head(20),
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Feature Importance")
plt.tight_layout()
plt.show()

print("\nTop Features")
print(feature_importance.head(20))

# ============================================================
# FINAL TEST PREDICTIONS (24K TEST SET)
# ============================================================

test_probs = final_model.predict_proba(
    X_test_final
)[:, 1]

test_predictions = (
    test_probs >= best_threshold
).astype(int)

submission = pd.DataFrame({
    "Prediction": test_predictions,
    "Probability": test_probs
})

submission.to_csv(
    "catboost_churn_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "catboost_churn_predictions.csv"
)