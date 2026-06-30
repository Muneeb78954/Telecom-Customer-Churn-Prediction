import pandas as pd
import numpy as np

train = pd.read_csv('training_dataset.csv')
test = pd.read_csv('testing_dataset.csv')

print('TRAIN SHAPE:', train.shape)
print('TEST SHAPE:', test.shape)
print('\nCLASS BALANCE (TRAIN):')
print(train['churn'].value_counts())
print('\nCLASS BALANCE RATIO:', train['churn'].value_counts(normalize=True).to_dict())
print('\nTRAIN COLUMNS:', train.columns.tolist())
print('\nMISSING VALUES:')
print(train.isnull().sum()[train.isnull().sum() > 0])
print('\nFEATURE CORRELATION WITH CHURN:')
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlations = train[numeric_cols].corr()['churn'].sort_values(ascending=False)
print(correlations.head(10))
