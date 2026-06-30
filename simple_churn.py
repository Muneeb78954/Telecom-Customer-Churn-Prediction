"""
Simple Churn Prediction with Key Improvements
==============================================
Tests: SMOTE, Class Weights, Better Thresholds, Ensemble
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

# Write output to file
log_file = open('model_run.log', 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("\n" + "="*60)
log("CHURN PREDICTION - KEY IMPROVEMENTS")
log("="*60)

try:
    log("\n[1/4] Loading data...")
    train = pd.read_csv('training_dataset.csv')
    test = pd.read_csv('testing_dataset.csv')
    y_train = train['churn'].copy()
    X_train = train.drop(columns=['churn'])
    X_test = test.copy()
    
    log(f"  Train: {train.shape} | Test: {test.shape}")
    log(f"  Churn rate: {train['churn'].mean():.2%}")
    
    # Lightweight feature engineering
    log("\n[2/4] Basic feature engineering...")
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Date features
        if 'date_of_registration' in df.columns:
            dor = pd.to_datetime(df['date_of_registration'], errors='coerce')
            df['reg_year'] = dor.dt.year
            df['reg_month'] = dor.dt.month
            df.drop(columns=['date_of_registration'], inplace=True)
        
        # Tenure segments
        tenure = df['customer_tenure_days'].clip(lower=1)
        df['is_new'] = (tenure < 90).astype(int)
        df['is_risky'] = ((tenure >= 30) & (tenure < 180)).astype(int)
        
        # Usage aggregates
        df['total_usage'] = df['calls_made'] + df['sms_sent'] + df['data_used']
        df['daily_engage'] = df['calls_per_day'] + df['sms_per_day'] + df['data_per_day']
        
        # Interactions
        df['tenure_x_activity'] = tenure * df['activity_score']
        df['age_x_usage'] = df['age'] * df['daily_engage']
        
        # Log transforms
        for col in ['estimated_salary', 'calls_made', 'customer_tenure_days']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    log(f"  Features created: {X_train.shape[1]}")
    
    # Preprocessing
    log("\n[3/4] Preprocessing...")
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    log(f"  Preprocessing complete")
    
    # Simple CatBoost with class weights
    log("\n[4/4] Training CatBoost with improvements...")
    from catboost import CatBoostClassifier
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        log(f"  Fold {fold}/5...", )
        
        X_tr, X_val = X_train_imp[tr_idx], X_train_imp[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        
        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test_imp)[:, 1] / 5
        
        # Find best threshold for this fold
        best_f1 = 0
        for t in np.arange(0.2, 0.8, 0.05):
            f1 = f1_score(y_val, (oof_pred[val_idx] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        
        log(f" Best F1={best_f1:.4f}")
    
    # Threshold optimization
    log("\n  Optimizing threshold...")
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.15, 0.85, 0.01):
        preds = (oof_pred >= threshold).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    log(f"\n  BEST F1 SCORE: {best_f1:.4f}")
    log(f"  BEST THRESHOLD: {best_threshold:.3f}")
    
    # Generate submission
    final_pred = (test_pred >= best_threshold).astype(int)
    submission = pd.DataFrame({
        'id': np.arange(1, len(test) + 1),
        'churn': final_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    log(f"\n  Submission saved!")
    log(f"  Predicted churned: {final_pred.sum()} ({final_pred.mean():.2%})")
    
    log("\n" + "="*60)
    log("SUCCESS - Check submission.csv")
    log("="*60 + "\n")

except Exception as e:
    log(f"\nERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())

finally:
    log_file.close()
