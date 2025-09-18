```python
import pandas as pd
import statsmodels.api as sm

# List of numerical confounders
confounders = [
    'SIZE', 'LEV', 'lctat', 'lctlt', 'chlct', 'CH', 'CF', 'Inventory',
    'Fixed', 'Growth', 'NI', 'boardmeetings', 'female', 'G_score', 'E_score',
    'gdpgrowth', 'gdppercapita', 'INF', 'n_years'
]

# Ensure no missing values (optional)
df.dropna(subset=['FV', 'ClimateRiskScore', 'ESG'] + confounders, inplace=True)
```

### ✅ Case 1: Direct effect (no confounders)
```python
X = df[['ClimateRiskScore']]
y = df['FV']
X = sm.add_constant(X)
model1 = sm.OLS(y, X).fit()
print(model1.summary())
```
1. Use Random Forest and XGB as baseline, with XGB using Optuna to optimize the params.


### ✅ Case 2: With numerical confounders
```python
X = df[['ClimateRiskScore'] + confounders]
X = sm.add_constant(X)
model2 = sm.OLS(y, X).fit()
print(model2.summary())
```


### ✅ Case 3: Moderating effect (ESG), no confounders
```python
X = df[['ClimateRiskScore', 'ESG']]
X['Interaction'] = X['ClimateRiskScore'] * X['ESG']
X = sm.add_constant(X)
model3 = sm.OLS(y, X).fit()
print(model3.summary())
```

### ✅ Case 4: Moderating effect with confounders
```python
X = df[['ClimateRiskScore', 'ESG'] + confounders]
X['Interaction'] = X['ClimateRiskScore'] * X['ESG']
X = sm.add_constant(X)
model4 = sm.OLS(y, X).fit()
print(model4.summary())
```


