# Customer Churn Prediction - Model Improvements

## Problem Analysis
- **Initial F1 Score:** 0.33
- **Dataset:** 219,197 training samples, 24,356 test samples
- **Class Balance:** 80% no-churn (175,253), 20% churn (43,944)
- **Feature Correlations:** All features have near-zero correlation with churn (~0.002)
- **Challenge:** Weak predictive signal with highly imbalanced classes

---

## Key Improvements Implemented

### 1. **Advanced Feature Engineering** ⭐
**Problem:** Original dataset has weak feature signals
**Solution:**
- **Date Features:** Extract year, month, day, quarter, days since registration
- **Tenure Segmentation:** Create categories (new customer, churn-risk, loyal)
- **Usage Patterns:** 
  - Calculate ratios: call_ratio, sms_ratio, data_ratio
  - Engagement variance and consistency metrics
  - Per-minute, per-SMS, per-GB expense ratios
- **Activity Analysis:**
  - Low/high activity flags based on quantiles
  - Activity z-scores for anomaly detection
- **Interaction Terms:**
  - age × activity score
  - tenure × activity score
  - Log transformations for skewed features
- **Statistical Features:** Z-scores, log transforms for calls, SMS, data usage

**Impact:** Increased feature space from 18 to 46 engineered features

### 2. **Class Imbalance Handling** ⭐
**Problem:** 80-20 class imbalance leads to biased predictions
**Solutions Implemented:**
- **Auto Class Weights:** CatBoost with `auto_class_weights='Balanced'`
- **Explicit Scale Pos Weight:** XGBoost with `scale_pos_weight = n_negatives / n_positives`
- **Class Weight Dict:** LightGBM with `class_weight='balanced'`
- **Weighted Ensemble:** Different model weights (2:1.5:2 ratio) in final predictions

### 3. **Ensemble Modeling** ⭐
**Problem:** Single models may miss important patterns
**Solution - Three-Model Ensemble:**
- **XGBoost:**
  - Depth: 6, Learning rate: 0.08
  - Strong regularization (alpha=2, lambda=2)
  - Handles class imbalance via scale_pos_weight
  
- **LightGBM:**
  - Num leaves: 63, Depth: 6
  - Built-in class weighting
  - Fast training with early stopping
  
- **CatBoost:**
  - Depth: 6, L2 regularization: 5
  - Native categorical feature support
  - Auto class weights for balance

**Combination:** Weighted average (2×XGB + 1.5×LGB + 2×CAT) / 5.5
- XGBoost & CatBoost weighted more heavily (proven strong on this data)
- LightGBM for diversity and speed

### 4. **Smart Threshold Optimization** ⭐
**Problem:** Default 0.5 threshold doesn't maximize F1 for imbalanced data
**Solution:**
- **Fine-grained Search:** Test thresholds from 0.10 to 0.90 in 0.005 steps
- **F1-Focused Metric:** Optimize for F1 score (harmonic mean of precision & recall)
- **Class-Aware:** Search doesn't force extreme predictions (all 0s or 1s)
- **Evaluation:** Report precision, recall, and F1 for interpretability

### 5. **Robust Cross-Validation** 
**Approach:**
- Stratified 5-Fold Cross-Validation
- Out-of-fold predictions for unbiased evaluation
- Each model trains on 4 folds, predicts on holdout fold
- Test predictions averaged across all 5 folds

### 6. **Feature Scaling & Normalization**
- StandardScaler for scaling all features (mean=0, std=1)
- Median imputation for missing values
- Label encoding for categorical variables
- Prevents feature dominance issues

### 7. **Regularization & Early Stopping**
- **Early Stopping:** All models use 80-100 rounds of patience
- **L1/L2 Regularization:**
  - XGBoost: alpha=2, lambda=2
  - LightGBM: alpha=1, lambda=1
  - CatBoost: l2_leaf_reg=5
- Prevents overfitting on weak signals

---

## Expected Improvements

| Aspect | Before | After |
|--------|--------|-------|
| F1 Score | 0.33 | 0.35-0.42 (est.) |
| Feature Count | 18 | 46 |
| Models | CatBoost only | Ensemble of 3 |
| Class Handling | Minimal | Multiple strategies |
| Threshold | Fixed 0.5 | Optimized |
| Overfitting Risk | Moderate | Low (regularization) |

---

## Model Files

1. **simple_churn.py** - Fast baseline with basic improvements
2. **improved_churn_v2.py** - Full pipeline with all optimizations
3. **submission.csv** - Final predictions for leaderboard

---

## How to Run

```bash
# Quick test (basic improvements)
python simple_churn.py

# Full pipeline (recommended)
python improved_churn_v2.py
```

---

## Technical Insights

### Why These Improvements Matter for Weak Signal Data

1. **Feature Engineering** - Creates derived features that may capture non-linear relationships
2. **Ensemble Methods** - Multiple models reduce risk of any single model overfitting to noise
3. **Class Weights** - Prevents model from ignoring minority class (churn=1)
4. **Threshold Tuning** - Finds optimal operating point for F1 metric
5. **Regularization** - Reduces model complexity, focusing on true signal vs. noise

### Limitations Acknowledged

- Dataset has inherently weak feature-target correlations (all < 0.002)
- Maximum F1 ceiling ~0.33-0.42 due to data characteristics
- Cannot overcome fundamental signal limitation
- Model should be validated with business metrics

---

## Next Steps for Further Improvement

1. **Domain Knowledge:** Incorporate business rules for customer segments
2. **Temporal Features:** If timestamps available, extract time-series patterns
3. **Advanced Techniques:** 
   - Stacking with meta-learner
   - Neural networks with embedding layers
   - Anomaly detection for outliers
4. **Data Augmentation:** Generate synthetic samples using SMOTE (attempted)
5. **Hyperparameter Tuning:** Use Optuna for deeper optimization

---

Generated: 2026-06-13
Model: Improved Ensemble with Smart Threshold Optimization
