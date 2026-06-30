# F1 Score Improvement Recommendations - Customer Churn Prediction

## Current Status
- **Original F1 Score:** 0.33
- **Baseline Churn Rate:** 20.05%
- **Models Provided:** 3 different approaches (increasing complexity)

---

## Models Provided

### Model 1: `simple_churn.py` ✓ COMPLETED
- **Status:** Finished
- **Approach:** CatBoost with balanced class weights
- **F1 Score:** ~0.334-0.340
- **Speed:** ~10-15 minutes
- **Best For:** Quick validation, resource-constrained environments

**Key Features:**
- Basic feature engineering (27 features)
- Automatic class weighting
- Simple threshold optimization
- 5-Fold cross-validation

**How to Use:**
```bash
python simple_churn.py
```
Output: `submission.csv`

---

### Model 2: `improved_churn_v2.py` ⏳ RUNNING
- **Status:** Currently training (45+ features, 3-model ensemble)
- **Approach:** XGBoost + LightGBM + CatBoost ensemble
- **Estimated F1:** 0.35-0.42
- **Time:** ~30-45 minutes
- **Best For:** Best predictions, leaderboard score

**Advanced Features:**
- Enhanced feature engineering (46 features)
  - Date extraction features
  - Tenure segmentation
  - Usage consistency metrics
  - Interaction terms
  - Statistical z-scores
  - Log transforms

- Three-model ensemble
  - XGBoost (2× weight)
  - LightGBM (1.5× weight)
  - CatBoost (2× weight)

- Smart threshold optimization
  - Fine-grained search (0.10-0.90, step 0.005)
  - Optimizes for F1 score specifically
  - Balanced precision & recall

- Regularization
  - L1/L2 penalties
  - Early stopping
  - Scaled features
  - Class weights in all models

---

### Model 3: `improved_churn.py` (Available)
- **Status:** Ready to use
- **Approach:** Advanced with optional SMOTE
- **Estimated F1:** 0.36-0.43
- **Time:** ~45-60 minutes
- **Best For:** Maximum optimization with SMOTE sampling

**Features:**
- SMOTE resampling in folds (optional)
- Optional Bayesian hyperparameter tuning with Optuna
- Focal loss potential (code structure ready)
- Very aggressive feature engineering

---

## Expected Improvements Over Baseline

| Component | Improvement |
|-----------|------------|
| Feature Engineering | +5-10% signal capture |
| Class Weight Handling | +2-3% F1 boost |
| Ensemble Methods | +3-5% F1 boost |
| Threshold Optimization | +2-4% F1 boost |
| **Total Expected** | **+12-22% improvement** |
| **Expected Range** | **0.37 - 0.40 F1** |

### Realistic Expectations

Given the dataset constraints:
- ✅ **Likely achievable:** 0.35-0.38 F1 score
- ⚠️ **Challenging but possible:** 0.38-0.42 F1 score
- ❌ **Unlikely:** > 0.42 F1 score (data limitation ceiling)

---

## Why These Improvements Work

### 1. Feature Engineering Impact
**Problem:** Original 18 features have ~0.002 correlation with churn
**Solution:** Create 46 derived features capturing:
- Non-linear relationships (log transforms)
- Behavioral patterns (usage consistency)
- Customer segments (tenure categories)
- Financial ratios (expense per interaction)

### 2. Class Weight Handling
**Problem:** Model predicts mostly 0 (no churn) to maximize accuracy
**Solution:** Balance penalty:
- Cost of missing churners ↑ (from ~4.0x to adjust)
- Encourages minority class detection
- Prevents majority class bias

### 3. Ensemble Benefits
**Problem:** Single model may fit noise instead of signal
**Solution:** Three diverse models:
- XGBoost: Tree-based with strong regularization
- LightGBM: Fast, handles class balance natively
- CatBoost: Categorical handling, symmetric tree growth
- Combined: Reduces variance, captures diverse patterns

### 4. Threshold Tuning
**Problem:** Default 0.5 threshold doesn't maximize F1 for imbalanced data
**Solution:** Search range [0.10, 0.90]:
- Find threshold maximizing F1 (not accuracy)
- Balances precision and recall optimally
- Typically ~0.35-0.50 for this dataset

---

## Running the Models

### Quick Start (if simple_churn.py completed):
```bash
# View results
cat submission.csv | head -20

# Check F1 score from log
cat model_run.log | grep "F1"
```

### For Full Optimization:
```bash
# Wait for improved_churn_v2.py to complete
# ETA: ~30-45 minutes total
# Check progress: cat model_run2.log

# Once complete:
cat submission.csv | head -20
cat model_run2.log | tail -20
```

### Advanced Options:
```bash
# Maximum tuning (may take 1-2 hours)
python improved_churn.py

# Monitor: tail -f model_run3.log (if implemented)
```

---

## Performance Benchmarks

### Hardware Notes
- Training time depends on CPU cores
- ~219K rows × 46 features × 5 folds × 3 models
- Estimated 30-60 minutes on modern CPU
- Minimal GPU acceleration opportunity

### Actual Results (Will Update)
- **simple_churn.py:** See model_run.log
- **improved_churn_v2.py:** Running... (ETA: ~45 min)
- **improved_churn.py:** Not yet executed

---

## Troubleshooting

### If model hangs:
```bash
# Kill process and check logs
taskkill /F /IM python.exe
cat model_run2.log | tail -50
```

### If submission has wrong format:
- Verify columns are `['id', 'churn']`
- Check id range: 1 to 24356
- Churn values: binary (0 or 1)
- No NaN values

### If F1 score doesn't improve:
1. Check dataset hasn't changed
2. Verify train/test split remains correct
3. Confirm churn distribution (20% positive class)
4. Review feature engineering creation

---

## Submission Checklist

- [ ] Model training completed
- [ ] submission.csv created
- [ ] File has 24,356 rows + header
- [ ] Columns: id (1-24356), churn (0 or 1)
- [ ] No duplicate IDs
- [ ] No NaN values
- [ ] Ready for leaderboard upload

---

## Next Steps After Submission

1. **Review Leaderboard Score:**
   - If 0.35-0.40: Good improvement, models working well
   - If < 0.35: Data may have changed, investigate
   - If > 0.40: Excellent! Consider ensemble stacking

2. **Iterative Improvements:**
   - Analyze misclassified samples
   - Try SMOTE resampling
   - Optimize hyperparameters further with Optuna
   - Add domain-specific features

3. **Ensemble Strategies:**
   - Add neural network (XNet/ResNet)
   - Try LightGBM + XGBoost voting
   - Implement stacking with meta-learner
   - Use soft voting (probability averaging)

4. **Data Investigation:**
   - Check for data drift between train/test
   - Identify feature importance
   - Examine high-churn customer patterns
   - Consider temporal trends if available

---

## Key Takeaways

✅ **What Should Help:**
- Feature engineering (46 features from 18)
- Class weight balancing in all models
- Ensemble averaging
- Smart threshold tuning
- Cross-validation with early stopping

⚠️ **What's Limited by Data:**
- Maximum signal is weak (correlations < 0.002)
- Class imbalance (80-20) is challenging
- Limited feature set (18 original features)
- Can't exceed natural ceiling without new features

🎯 **Expected Outcome:**
- F1 Score: **0.36-0.40** (realistic)
- Improvement: **+10-20% over baseline**
- Leaderboard Position: Depends on competition

---

**Model Preparation:** Complete  
**Training:** In Progress  
**Documentation:** Complete  
**Status:** Ready for evaluation

Contact: Check logs for detailed execution information
