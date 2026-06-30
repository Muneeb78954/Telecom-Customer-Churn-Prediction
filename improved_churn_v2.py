"""
Improved Churn Prediction with Better Optimization
===================================================
Key improvements:
1. Better feature engineering with statistical features
2. Multiple models ensemble
3. Smart threshold optimization that respects class balance
4. Early stopping and regularization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.impute import SimpleImputer

log_file = open('model_run2.log', 'w')

def log(msg=''):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("\n" + "="*70)
log("IMPROVED CHURN PREDICTION MODEL")
log("="*70)

try:
    log("\n[1/5] Loading & EDA...")
    train = pd.read_csv('training_dataset.csv')
    test = pd.read_csv('testing_dataset.csv')
    y_train = train['churn'].copy()
    
    log(f"Train: {train.shape} | Test: {test.shape}")
    log(f"Churn rate: {y_train.mean():.2%}")
    log(f"Class distribution - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    
    # ──────────────────────────────────────────────────────────
    # ENHANCED FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────
    log("\n[2/5] Feature Engineering...")
    
    def create_features(df):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Date extraction
        if 'date_of_registration' in df.columns:
            dor = pd.to_datetime(df['date_of_registration'], errors='coerce')
            df['reg_year'] = dor.dt.year
            df['reg_month'] = dor.dt.month
            df['reg_day_of_month'] = dor.dt.day
            df['reg_quarter'] = dor.dt.quarter
            df['days_since_reg'] = (pd.Timestamp.now() - dor).dt.days
            df.drop(columns=['date_of_registration'], inplace=True)
        
        # Tenure categorization
        tenure = df['customer_tenure_days'].clip(lower=1)
        df['is_new_customer'] = (tenure < 30).astype(int)
        df['is_churning_risk'] = ((tenure >= 30) & (tenure < 180)).astype(int)
        df['is_loyal_customer'] = (tenure > 365).astype(int)
        df['tenure_months'] = tenure / 30
        
        # Usage analysis
        df['total_usage'] = df['calls_made'] + df['sms_sent'] + df['data_used']
        df['call_ratio'] = df['calls_made'] / (df['total_usage'] + 1)
        df['sms_ratio'] = df['sms_sent'] / (df['total_usage'] + 1)
        df['data_ratio'] = df['data_used'] / (df['total_usage'] + 1)
        
        # Daily engagement metrics
        df['daily_total'] = df['calls_per_day'] + df['sms_per_day'] + df['data_per_day']
        df['engagement_variance'] = np.abs(df['calls_per_day'] - df['sms_per_day'] - df['data_per_day'])
        
        # Usage consistency
        df['usage_consistency'] = df['daily_total'] / (df['total_usage'] / (tenure + 1) + 1)
        
        # Age features
        df['age_group'] = pd.cut(df['age'], bins=[0,20,30,40,50,60,100], labels=[0,1,2,3,4,5]).astype(int)
        
        # Financial metrics
        df['expense_per_minute_talk'] = df['estimated_salary'] / (df['calls_made'] * 2 + 1)
        df['expense_per_sms'] = df['estimated_salary'] / (df['sms_sent'] + 1)
        df['expense_per_gb_data'] = df['estimated_salary'] / (df['data_used'] + 1)
        
        # Activity score analysis
        df['low_activity'] = (df['activity_score'] < df['activity_score'].quantile(0.25)).astype(int)
        df['high_activity'] = (df['activity_score'] > df['activity_score'].quantile(0.75)).astype(int)
        
        # Interaction features
        df['age_activity_interaction'] = df['age'] * df['activity_score'] / 1000
        df['tenure_activity_interaction'] = tenure * df['activity_score'] / 1000
        
        # Statistical features
        for col in ['calls_made', 'sms_sent', 'data_used']:
            if col in df.columns:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1)
                df[f'log_{col}'] = np.log1p(df[col])
        
        # Fill any remaining NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    X_train = create_features(train.drop(columns=['churn']))
    X_test = create_features(test)
    
    log(f"Features: {X_train.shape[1]}")
    
    # ──────────────────────────────────────────────────────────
    # PREPROCESSING
    # ──────────────────────────────────────────────────────────
    log("\n[3/5] Preprocessing...")
    
    # Categorical encoding
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    log("Preprocessing complete")
    
    # ──────────────────────────────────────────────────────────
    # MODEL TRAINING WITH ENSEMBLE
    # ──────────────────────────────────────────────────────────
    log("\n[4/5] Training Models...")
    
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize predictions
    xgb_oof = np.zeros(len(X_train))
    lgb_oof = np.zeros(len(X_train))
    cat_oof = np.zeros(len(X_train))
    
    xgb_test = np.zeros(len(X_test))
    lgb_test = np.zeros(len(X_test))
    cat_test = np.zeros(len(X_test))
    
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        log(f"  Fold {fold}...", )
        
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx].values, y_train.iloc[val_idx].values
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.08, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=2, reg_lambda=2,
            scale_pos_weight=class_weight[1],
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_oof[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        xgb_test += xgb_model.predict_proba(X_test_scaled)[:, 1] / 5
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            max_depth=6, learning_rate=0.08, n_estimators=500,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1, reg_lambda=1, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(-1)
        ])
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        lgb_test += lgb_model.predict_proba(X_test_scaled)[:, 1] / 5
        
        # CatBoost
        cat_model = CatBoostClassifier(
            depth=6, learning_rate=0.08, iterations=500,
            l2_leaf_reg=5, random_seed=42, verbose=False,
            auto_class_weights='Balanced'
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=80, verbose=False)
        cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        cat_test += cat_model.predict_proba(X_test_scaled)[:, 1] / 5
        
        log("done")
    
    # ──────────────────────────────────────────────────────────
    # ENSEMBLE & THRESHOLD OPTIMIZATION
    # ──────────────────────────────────────────────────────────
    log("\n[5/5] Optimization...")
    
    # Weighted ensemble
    ensemble_oof = (2*xgb_oof + 1.5*lgb_oof + 2*cat_oof) / 5.5
    ensemble_test = (2*xgb_test + 1.5*lgb_test + 2*cat_test) / 5.5
    
    # Find best threshold using F1 score
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for threshold in np.arange(0.10, 0.90, 0.005):
        preds = (ensemble_oof >= threshold).astype(int)
        f1 = f1_score(y_train.values, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            precision = (preds & y_train.values).sum() / (preds.sum() + 1)
            recall = (preds & y_train.values).sum() / (y_train.values.sum() + 1)
            best_precision = precision
            best_recall = recall
    
    log(f"\n  Best Threshold: {best_threshold:.3f}")
    log(f"  OOF F1 Score: {best_f1:.4f}")
    log(f"  Precision: {best_precision:.4f}")
    log(f"  Recall: {best_recall:.4f}")
    
    # Generate submission
    final_pred = (ensemble_test >= best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'id': np.arange(1, len(test) + 1),
        'churn': final_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    
    log(f"\n  Submission saved!")
    log(f"  Predicted churned: {final_pred.sum()} ({final_pred.mean():.2%})")
    log(f"  Predicted retained: {(1-final_pred).sum()} ({(1-final_pred).mean():.2%})")
    
    log("\n" + "="*70)
    log("SUCCESS!")
    log("="*70)

except Exception as e:
    log(f"\nERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())

finally:
    log_file.close()
