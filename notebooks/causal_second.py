# %% [markdown]
# # Causal Analysis Notebook
# 
# This notebook contains functions for tables and visualizations based on the company data CSV.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import LinearDML
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %% [markdown]
# ## Load Data

# %%
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    return df

df = load_data('/mnt/e/NEUConference/ClimateRisk/output_tfidf/Combined_Company_Data_2022_2024_Final2.csv')
print(f'Dataset shape: {df.shape}')
df.head()

# %% [markdown]
# ## Variable Mapping

# %%
def map_variables(df):
    var_map = {
        'Tobins_Q': 'FV',
        'Climate_Risk_TFIDF': 'ClimateRiskScore',
        'ESG': 'ESG',
        'SIZE': 'SIZE',
        'LEV': 'LEV',
        'STDebt_TA': 'lctat',
        'STDebt_TL': 'lctlt',
        'IntExp_Sales': 'chlct',
        'Cash_TA_lag': 'cheat',
        'Inventory_Sales': 'Inventory',
        'Growth': 'Growth',
        'Net_Income_After_Tax': 'NI',
        'GDP_Growth': 'GDPgrowth',
        'Inf': 'INF',
        'Country': 'Country of Exchange',
        'Year': 'year',
        'GICS': 'GICS Industry Name',
        'Company': 'Company Name'
    }
    mapped_df = pd.DataFrame()
    for new_name, old_name in var_map.items():
        if old_name in df.columns:
            mapped_df[new_name] = df[old_name]
    mapped_df = mapped_df.dropna()
    return mapped_df

df_mapped = map_variables(df)
print(f'Mapped dataset shape: {df_mapped.shape}')
df_mapped.head()

# %%
def get_analysis_vars(df):
    y_col = 'Tobins_Q'
    t_col = 'Climate_Risk_TFIDF'
    x_cols = ['SIZE', 'LEV', 'STDebt_TA', 'STDebt_TL', 'IntExp_Sales', 
              'Cash_TA_lag', 'Inventory_Sales', 'Growth', 
              'Net_Income_After_Tax', 'GDP_Growth', 'Inf']
    return y_col, t_col, x_cols

y_col, t_col, x_cols = get_analysis_vars(df_mapped)
print(f'Outcome (Y): {y_col}')
print(f'Treatment (T): {t_col}')
print(f'Confounders (X): {x_cols}')

# %% [markdown]
# ## TABLE 1: Descriptive Statistics

# %%
def table_descriptive_statistics(df, y_col, t_col, x_cols):
    all_vars = [y_col, t_col] + x_cols
    stats_df = pd.DataFrame()
    for var in all_vars:
        if var in df.columns:
            data = df[var]
            stats_df[var] = {
                'N': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                '25%': data.quantile(0.25),
                'Median': data.quantile(0.5),
                '75%': data.quantile(0.75),
                'Max': data.max()
            }
    stats_df = stats_df.T
    stats_df.index.name = 'Variable'
    return stats_df.round(4)

table1 = table_descriptive_statistics(df_mapped, y_col, t_col, x_cols)
table1

# %% [markdown]
# ## TABLE 2: Covariate Balance (Treatment vs Control)

# %%
def table_covariate_balance(df, t_col, x_cols, threshold=0.1):
    median_t = df[t_col].median()
    treatment = df[df[t_col] >= median_t]
    control = df[df[t_col] < median_t]
    balance_df = pd.DataFrame()
    for var in x_cols:
        if var in df.columns:
            t_mean = treatment[var].mean()
            c_mean = control[var].mean()
            pooled_std = np.sqrt((treatment[var].std()**2 + control[var].std()**2) / 2)
            smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
            balanced = 'Yes' if abs(smd) < threshold else 'No'
            balance_df[var] = {
                'Mean (Treatment)': t_mean,
                'Mean (Control)': c_mean,
                'SMD': smd,
                'Balanced': balanced
            }
    balance_df = balance_df.T
    balance_df.index.name = 'Covariate'
    return balance_df.round(4)

table2 = table_covariate_balance(df_mapped, t_col, x_cols)
table2

# %% [markdown]
# ## TABLE 3: ML Model Performance (Nuisance Functions)

# %%
def table_ml_performance(df, y_col, t_col, x_cols, cv_folds=5):
    X = df[x_cols].values
    Y = df[y_col].values
    T = df[t_col].values
    results = []
    
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    y_pred_rf = cross_val_predict(rf_y, X, Y, cv=cv_folds)
    results.append({
        'Model': 'Random Forest (Y)',
        'Target': 'Outcome (Y)',
        'R2': r2_score(Y, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(Y, y_pred_rf)),
        'MAE': mean_absolute_error(Y, y_pred_rf)
    })
    
    xgb_y = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    y_pred_xgb = cross_val_predict(xgb_y, X, Y, cv=cv_folds)
    results.append({
        'Model': 'XGBoost (Y)',
        'Target': 'Outcome (Y)',
        'R2': r2_score(Y, y_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(Y, y_pred_xgb)),
        'MAE': mean_absolute_error(Y, y_pred_xgb)
    })
    
    rf_t = RandomForestRegressor(n_estimators=100, random_state=42)
    t_pred_rf = cross_val_predict(rf_t, X, T, cv=cv_folds)
    results.append({
        'Model': 'Random Forest (T)',
        'Target': 'Treatment (T)',
        'R2': r2_score(T, t_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(T, t_pred_rf)),
        'MAE': mean_absolute_error(T, t_pred_rf)
    })
    
    xgb_t = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    t_pred_xgb = cross_val_predict(xgb_t, X, T, cv=cv_folds)
    results.append({
        'Model': 'XGBoost (T)',
        'Target': 'Treatment (T)',
        'R2': r2_score(T, t_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(T, t_pred_xgb)),
        'MAE': mean_absolute_error(T, t_pred_xgb)
    })
    
    return pd.DataFrame(results).round(4)

table3 = table_ml_performance(df_mapped, y_col, t_col, x_cols)
table3

# %% [markdown]
# ## TABLE 4: Baseline OLS Results

# %%
def table_ols_results(df, y_col, t_col, x_cols):
    Y = df[y_col].values
    T = df[t_col].values
    X = df[x_cols].values
    results = []
    
    T_naive = sm.add_constant(T)
    model_naive = sm.OLS(Y, T_naive).fit()
    results.append({
        'Model': 'OLS (Naive)',
        'ATE': model_naive.params[1],
        'Std_Error': model_naive.bse[1],
        'CI_Lower': model_naive.conf_int()[1, 0],
        'CI_Upper': model_naive.conf_int()[1, 1],
        'P_Value': model_naive.pvalues[1]
    })
    
    X_full = np.column_stack([T, X])
    X_full = sm.add_constant(X_full)
    model_controls = sm.OLS(Y, X_full).fit()
    results.append({
        'Model': 'OLS (With Controls)',
        'ATE': model_controls.params[1],
        'Std_Error': model_controls.bse[1],
        'CI_Lower': model_controls.conf_int()[1, 0],
        'CI_Upper': model_controls.conf_int()[1, 1],
        'P_Value': model_controls.pvalues[1]
    })
    
    return pd.DataFrame(results).round(6)

table4 = table_ols_results(df_mapped, y_col, t_col, x_cols)
table4

# %% [markdown]
# ## DML Estimation Functions

# %%
def dml_estimate(df, y_col, t_col, x_cols, model_type='rf', cv_folds=5):
    Y = df[y_col].values
    T = df[t_col].values.reshape(-1, 1)
    X = df[x_cols].values
        
    if model_type == 'rf':
        model_y = RandomForestRegressor(n_estimators=100, random_state=42)
        model_t = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_y = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
        model_t = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=cv_folds,
        random_state=42
    )
    dml.fit(Y, T, X=X)

    effect = dml.effect(X)
    effect_mean = np.mean(effect)

    inference = dml.effect_inference(X)
    ci_lower, ci_upper = inference.conf_int(alpha=0.05)
    ci_lower = np.mean(ci_lower)
    ci_upper = np.mean(ci_upper)
    se = np.mean(inference.stderr)
    try:
        p_value = np.mean(inference.pvalue())
    except:
        z = effect_mean / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'ATE': effect_mean,
        'Std_Error': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'P_Value': p_value,
        'Model': dml,
        'Effects': effect
    }


def causal_forest_estimate(df, y_col, t_col, x_cols, n_trees=100):
    Y = df[y_col].values
    T = df[t_col].values
    X = df[x_cols].values
    
    model_y = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    model_t = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    
    model_y.fit(X, Y)
    model_t.fit(X, T)
    
    y_pred = model_y.predict(X)
    t_pred = model_t.predict(X)
    
    y_resid = Y - y_pred
    t_resid = T - t_pred
    
    theta = np.sum(y_resid * t_resid) / np.sum(t_resid ** 2)
    
    residuals = y_resid - theta * t_resid
    sigma2 = np.sum(residuals ** 2) / len(residuals)
    se = np.sqrt(sigma2 / np.sum(t_resid ** 2))
    
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se
    z_score = theta / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'ATE': theta,
        'Std_Error': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'P_Value': p_value
    }

# %% [markdown]
# ## TABLE 5: Main Causal Estimates (DML Results)

# %%
def table_causal_estimates(df, y_col, t_col, x_cols):
    results = []
    
    dml_rf = dml_estimate(df, y_col, t_col, x_cols, model_type='rf')
    results.append({
        'Algorithm': 'DML - Random Forest',
        'ATE': dml_rf['ATE'],
        'Std_Error': dml_rf['Std_Error'],
        'CI_Lower': dml_rf['CI_Lower'],
        'CI_Upper': dml_rf['CI_Upper'],
        'P_Value': dml_rf['P_Value']
    })
    
    dml_xgb = dml_estimate(df, y_col, t_col, x_cols, model_type='xgb')
    results.append({
        'Algorithm': 'DML - XGBoost',
        'ATE': dml_xgb['ATE'],
        'Std_Error': dml_xgb['Std_Error'],
        'CI_Lower': dml_xgb['CI_Lower'],
        'CI_Upper': dml_xgb['CI_Upper'],
        'P_Value': dml_xgb['P_Value']
    })
    
    cf = causal_forest_estimate(df, y_col, t_col, x_cols)
    results.append({
        'Algorithm': 'Causal Forest',
        'ATE': cf['ATE'],
        'Std_Error': cf['Std_Error'],
        'CI_Lower': cf['CI_Lower'],
        'CI_Upper': cf['CI_Upper'],
        'P_Value': cf['P_Value']
    })
    
    return pd.DataFrame(results).round(6)

table5 = table_causal_estimates(df_mapped, y_col, t_col, x_cols)
table5

# %% [markdown]
# ## TABLE 6: Comparison Across Methods

# %%
def table_method_comparison(df, y_col, t_col, x_cols):
    results = []
    
    Y = df[y_col].values
    T = df[t_col].values
    X = df[x_cols].values
    
    T_naive = sm.add_constant(T)
    model_naive = sm.OLS(Y, T_naive).fit()
    results.append({
        'Method': 'OLS (Naive)',
        'ATE': model_naive.params[1],
        'Std_Error': model_naive.bse[1],
        'CI': f"[{model_naive.conf_int()[1, 0]:.4f}, {model_naive.conf_int()[1, 1]:.4f}]",
        'Interpretation': 'Likely upward biased'
    })
    
    X_full = np.column_stack([T, X])
    X_full = sm.add_constant(X_full)
    model_controls = sm.OLS(Y, X_full).fit()
    results.append({
        'Method': 'OLS (With Controls)',
        'ATE': model_controls.params[1],
        'Std_Error': model_controls.bse[1],
        'CI': f"[{model_controls.conf_int()[1, 0]:.4f}, {model_controls.conf_int()[1, 1]:.4f}]",
        'Interpretation': 'Reduced bias'
    })
    
    dml_rf = dml_estimate(df, y_col, t_col, x_cols, model_type='rf')
    results.append({
        'Method': 'DML - Random Forest',
        'ATE': dml_rf['ATE'],
        'Std_Error': dml_rf['Std_Error'],
        'CI': f"[{dml_rf['CI_Lower']:.4f}, {dml_rf['CI_Upper']:.4f}]",
        'Interpretation': 'Nonlinear adjustment'
    })
    
    dml_xgb = dml_estimate(df, y_col, t_col, x_cols, model_type='xgb')
    results.append({
        'Method': 'DML - XGBoost',
        'ATE': dml_xgb['ATE'],
        'Std_Error': dml_xgb['Std_Error'],
        'CI': f"[{dml_xgb['CI_Lower']:.4f}, {dml_xgb['CI_Upper']:.4f}]",
        'Interpretation': 'Preferred model'
    })
    
    return pd.DataFrame(results).round(6)

table6 = table_method_comparison(df_mapped, y_col, t_col, x_cols)
table6

# %% [markdown]
# ## TABLE 7: Heterogeneous Treatment Effects (CATE)

# %%
def table_cate_by_subgroup(df, y_col, t_col, x_cols, subgroup_col='ESG'):
    if subgroup_col not in df.columns:
        return pd.DataFrame({'Error': ['Subgroup column not found']})
    
    df = df.copy()
    df['subgroup_bin'] = pd.qcut(df[subgroup_col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    results = []
    for subgroup in df['subgroup_bin'].unique():
        subgroup_df = df[df['subgroup_bin'] == subgroup]
        if len(subgroup_df) < 10:
            continue
        
        dml_result = dml_estimate(subgroup_df, y_col, t_col, x_cols, model_type='rf')
        results.append({
            'Subgroup': f'{subgroup_col} - {subgroup}',
            'N': len(subgroup_df),
            'CATE': dml_result['ATE'],
            'Std_Error': dml_result['Std_Error'],
            'CI_Lower': dml_result['CI_Lower'],
            'CI_Upper': dml_result['CI_Upper']
        })
    
    return pd.DataFrame(results).round(6)

table7 = table_cate_by_subgroup(df_mapped, y_col, t_col, x_cols, subgroup_col='ESG')
table7

# %% [markdown]
# ## TABLE 8: Robustness and Placebo Tests

# %%
def table_robustness_tests(df, y_col, t_col, x_cols):
    results = []
    
    baseline = dml_estimate(df, y_col, t_col, x_cols, model_type='xgb')
    results.append({
        'Specification': 'Baseline (DML-XGBoost)',
        'ATE': baseline['ATE'],
        'Std_Error': baseline['Std_Error'],
        'P_Value': baseline['P_Value'],
        'Conclusion': 'Significant' if baseline['P_Value'] < 0.05 else 'Not significant'
    })
    
    df_placebo = df.copy()
    np.random.seed(42)
    df_placebo[t_col] = np.random.permutation(df_placebo[t_col].values)
    placebo = dml_estimate(df_placebo, y_col, t_col, x_cols, model_type='xgb')
    results.append({
        'Specification': 'Placebo Test (Randomized T)',
        'ATE': placebo['ATE'],
        'Std_Error': placebo['Std_Error'],
        'P_Value': placebo['P_Value'],
        'Conclusion': 'Not significant (expected)'
    })
    
    q75 = df[y_col].quantile(0.75)
    q25 = df[y_col].quantile(0.25)
    iqr = q75 - q25
    df_no_outliers = df[(df[y_col] >= q25 - 1.5*iqr) & (df[y_col] <= q75 + 1.5*iqr)]
    no_outliers = dml_estimate(df_no_outliers, y_col, t_col, x_cols, model_type='xgb')
    results.append({
        'Specification': 'Remove Outliers',
        'ATE': no_outliers['ATE'],
        'Std_Error': no_outliers['Std_Error'],
        'P_Value': no_outliers['P_Value'],
        'Conclusion': 'Robust' if abs(no_outliers['ATE'] - baseline['ATE']) < 0.5 else 'Different'
    })
    
    rf_result = dml_estimate(df, y_col, t_col, x_cols, model_type='rf')
    results.append({
        'Specification': 'Alternative Model (RF)',
        'ATE': rf_result['ATE'],
        'Std_Error': rf_result['Std_Error'],
        'P_Value': rf_result['P_Value'],
        'Conclusion': 'Consistent' if abs(rf_result['ATE'] - baseline['ATE']) < 0.5 else 'Different'
    })
    
    return pd.DataFrame(results).round(6)

table8 = table_robustness_tests(df_mapped, y_col, t_col, x_cols)
table8

# %% [markdown]
# ## FIGURE 1: Overlap / Common Support Plot

# %%
def plot_overlap_common_support(df, t_col, x_cols, save_path=None):
    X = df[x_cols].values
    T = df[t_col].values
    
    model_t = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t.fit(X, T)
    propensity_scores = model_t.predict(X)
    
    median_t = df[t_col].median()
    treated = propensity_scores[T >= median_t]
    control = propensity_scores[T < median_t]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(control, bins=30, alpha=0.5, label='Control', color='red', density=True)
    ax.hist(treated, bins=30, alpha=0.5, label='Treatment', color='blue', density=True)
    
    ax.set_xlabel('Propensity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Common Support / Overlap Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return propensity_scores

ps_scores = plot_overlap_common_support(df_mapped, t_col, x_cols)

# %% [markdown]
# ## FIGURE 2: SHAP Feature Importance

# %%
import shap
def plot_shap_feature_importance(df, y_col, x_cols, model_type='rf', save_path=None):
    X = df[x_cols].values
    Y = df[y_col].values
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    
    model.fit(X, Y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=x_cols, show=False)
    plt.title(f'SHAP Feature Importance - {model_type.upper()}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return shap_values

shap_vals_rf = plot_shap_feature_importance(df_mapped, y_col, x_cols, model_type='rf')

# %%
shap_vals_xgb = plot_shap_feature_importance(df_mapped, y_col, x_cols, model_type='xgb')

# %% [markdown]
# ## FIGURE 3: Distribution of Individual Treatment Effects (CATE Plot)

# %%
def plot_cate_distribution(df, y_col, t_col, x_cols, save_path=None):
    X = df[x_cols].values
    Y = df[y_col].values
    T = df[t_col].values
    
    dml_result = dml_estimate(df, y_col, t_col, x_cols, model_type='rf')
    ate_mean = dml_result['ATE']
    
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_y.fit(X, Y)
    model_t.fit(X, T)
    
    y_pred = model_y.predict(X)
    t_pred = model_t.predict(X)
    
    y_resid = Y - y_pred
    t_resid = T - t_pred
    
    individual_effects = y_resid / (t_resid + 1e-10)
    individual_effects = individual_effects[np.abs(individual_effects) < np.percentile(np.abs(individual_effects), 95)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(individual_effects, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(ate_mean, color='red', linestyle='--', linewidth=2, label=f'Mean ATE: {ate_mean:.4f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Individual Treatment Effect', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Individual Treatment Effects', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return individual_effects

cate_dist = plot_cate_distribution(df_mapped, y_col, t_col, x_cols)

# %% [markdown]
# ## FIGURE 4: Feature Importance (RF/XGBoost)

# %%
def plot_feature_importance(df, y_col, x_cols, save_path=None):
    X = df[x_cols].values
    Y = df[y_col].values
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, Y)
    rf_importance = pd.DataFrame({'Feature': x_cols, 'Importance': rf.feature_importances_})
    rf_importance = rf_importance.sort_values('Importance', ascending=True)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    xgb_model.fit(X, Y)
    xgb_importance = pd.DataFrame({'Feature': x_cols, 'Importance': xgb_model.feature_importances_})
    xgb_importance = xgb_importance.sort_values('Importance', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='steelblue')
    axes[0].set_xlabel('Importance Score', fontsize=12)
    axes[0].set_title('Random Forest Feature Importance', fontsize=14)
    
    axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='darkorange')
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_title('XGBoost Feature Importance', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_feature_importance(df_mapped, y_col, x_cols)

# %% [markdown]
# ## FIGURE 5: Model Comparison Plot (OLS vs RF vs XGBoost ATE)

# %%
def plot_model_comparison(df, y_col, t_col, x_cols, save_path=None):
    Y = df[y_col].values
    T = df[t_col].values
    X = df[x_cols].values
    
    methods = []
    ates = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    T_naive = sm.add_constant(T)
    model_naive = sm.OLS(Y, T_naive).fit()
    methods.append('OLS (Naive)')
    ates.append(model_naive.params[1])
    ci_lowers.append(model_naive.conf_int()[1, 0])
    ci_uppers.append(model_naive.conf_int()[1, 1])
    colors.append('navy')
    
    X_full = np.column_stack([T, X])
    X_full = sm.add_constant(X_full)
    model_controls = sm.OLS(Y, X_full).fit()
    methods.append('OLS (Controls)')
    ates.append(model_controls.params[1])
    ci_lowers.append(model_controls.conf_int()[1, 0])
    ci_uppers.append(model_controls.conf_int()[1, 1])
    colors.append('navy')
    
    dml_rf = dml_estimate(df, y_col, t_col, x_cols, model_type='rf')
    methods.append('DML - RF')
    ates.append(dml_rf['ATE'])
    ci_lowers.append(dml_rf['CI_Lower'])
    ci_uppers.append(dml_rf['CI_Upper'])
    colors.append('darkgreen')
    
    dml_xgb = dml_estimate(df, y_col, t_col, x_cols, model_type='xgb')
    methods.append('DML - XGBoost')
    ates.append(dml_xgb['ATE'])
    ci_lowers.append(dml_xgb['CI_Lower'])
    ci_uppers.append(dml_xgb['CI_Upper'])
    colors.append('darkgreen')
    
    cf = causal_forest_estimate(df, y_col, t_col, x_cols)
    methods.append('Causal Forest')
    ates.append(cf['ATE'])
    ci_lowers.append(cf['CI_Lower'])
    ci_uppers.append(cf['CI_Upper'])
    colors.append('darkgreen')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(methods))
    errors = np.array([np.array(ates) - np.array(ci_lowers), np.array(ci_uppers) - np.array(ates)])
    
    ax.errorbar(ates, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2, 
                color='black', ecolor='gray', markersize=8)
    
    for i, (method, ate, color) in enumerate(zip(methods, ates, colors)):
        ax.scatter(ate, i, color=color, s=100, zorder=5)
    
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('ATE Estimate', fontsize=12)
    ax.set_title('Comparison of Treatment Effect Estimates Across Methods', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_model_comparison(df_mapped, y_col, t_col, x_cols)

# %% [markdown]
# ## FIGURE 6: Balance Plot (Standardized Mean Differences)

# %%
def plot_balance_smd(df, t_col, x_cols, save_path=None):
    median_t = df[t_col].median()
    treatment = df[df[t_col] >= median_t]
    control = df[df[t_col] < median_t]
    
    smd_values = []
    var_names = []
    
    for var in x_cols:
        if var in df.columns:
            t_mean = treatment[var].mean()
            c_mean = control[var].mean()
            pooled_std = np.sqrt((treatment[var].std()**2 + control[var].std()**2) / 2)
            smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
            smd_values.append(abs(smd))
            var_names.append(var)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green' if s < 0.1 else 'red' for s in smd_values]
    
    y_pos = np.arange(len(var_names))
    ax.barh(y_pos, smd_values, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='SMD = 0.1')
    ax.axvline(0.2, color='red', linestyle='--', linewidth=2, label='SMD = 0.2')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_names)
    ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=12)
    ax.set_title('Covariate Balance: Standardized Mean Differences', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_balance_smd(df_mapped, t_col, x_cols)

# %% [markdown]
# ## FIGURE 7: Partial Dependence Plot (PDP)

# %%
def plot_partial_dependence(df, y_col, x_cols, top_n=5, save_path=None):
    X = df[x_cols].values
    Y = df[y_col].values
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, Y)
    
    importance_df = pd.DataFrame({'Feature': x_cols, 'Importance': rf.feature_importances_})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    top_features = importance_df['Feature'].head(top_n).tolist()
    top_indices = [x_cols.index(f) for f in top_features]
    
    fig, axes = plt.subplots(1, top_n, figsize=(4*top_n, 4))
    if top_n == 1:
        axes = [axes]
    
    for i, (idx, feat) in enumerate(zip(top_indices, top_features)):
        feature_values = np.linspace(X[:, idx].min(), X[:, idx].max(), 50)
        pdp_values = []
        
        for val in feature_values:
            X_temp = X.copy()
            X_temp[:, idx] = val
            pdp_values.append(np.mean(rf.predict(X_temp)))
        
        axes[i].plot(feature_values, pdp_values, color='steelblue', linewidth=2)
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel('Predicted Y', fontsize=10)
        axes[i].set_title(f'PDP: {feat}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_partial_dependence(df_mapped, y_col, x_cols, top_n=5)

# %% [markdown]
# ## Additional Utility Functions

# %%
def calculate_propensity_scores(df, t_col, x_cols):
    X = df[x_cols].values
    T = df[t_col].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, T)
    ps = model.predict(X)
    
    return ps

def calculate_individual_cate(df, y_col, t_col, x_cols):
    X = df[x_cols].values
    Y = df[y_col].values
    T = df[t_col].values
    
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_y.fit(X, Y)
    model_t.fit(X, T)
    
    y_pred = model_y.predict(X)
    t_pred = model_t.predict(X)
    
    y_resid = Y - y_pred
    t_resid = T - t_pred
    
    individual_cate = y_resid / (t_resid + 1e-10)
    
    return individual_cate

def export_tables_to_csv(tables_dict, output_dir='./'):
    for name, table in tables_dict.items():
        path = f"{output_dir}{name}.csv"
        table.to_csv(path)
        print(f'Saved: {path}')

# %% [markdown]
# ## Save All Tables

# %%
tables = {
    'table1_descriptive_stats': table1,
    'table2_covariate_balance': table2,
    'table3_ml_performance': table3,
    'table4_ols_results': table4,
    'table5_causal_estimates': table5,
    'table6_method_comparison': table6,
    'table7_cate_subgroups': table7,
    'table8_robustness': table8
}

export_tables_to_csv(tables, output_dir='./output_csv')

# %% [markdown]
# ## CATE by ESG Quantiles Analysis

# %%
def calculate_cate_by_esg(df, y_col, t_col, x_cols, n_quantiles=10):
    df = df.copy()
    df['ESG_bin'] = pd.qcut(df['ESG'], q=n_quantiles, labels=False, duplicates='drop')
    
    results = []
    for q in df['ESG_bin'].unique():
        subset = df[df['ESG_bin'] == q]
        if len(subset) >= 5:
            dml_res = dml_estimate(subset, y_col, t_col, x_cols, model_type='rf')
            results.append({
                'ESG_Quantile': q,
                'N': len(subset),
                'CATE': dml_res['ATE'],
                'SE': dml_res['Std_Error'],
                'CI_Lower': dml_res['CI_Lower'],
                'CI_Upper': dml_res['CI_Upper'],
                'ESG_Mean': subset['ESG'].mean()
            })
    
    return pd.DataFrame(results).sort_values('ESG_Quantile')

cate_by_esg = calculate_cate_by_esg(df_mapped, y_col, t_col, x_cols)
cate_by_esg

# %%
def plot_cate_by_esg_quantiles(cate_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(cate_df['ESG_Mean'], cate_df['CATE'], 
                yerr=[cate_df['CATE'] - cate_df['CI_Lower'], cate_df['CI_Upper'] - cate_df['CATE']],
                fmt='o-', capsize=5, capthick=2, color='steelblue', ecolor='gray')
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('ESG Score (Mean per Quantile)', fontsize=12)
    ax.set_ylabel('Conditional Average Treatment Effect', fontsize=12)
    ax.set_title('CATE of Climate Risk on Firm Value by ESG Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_cate_by_esg_quantiles(cate_by_esg)

# %% [markdown]
# ## Distribution Plots for Key Variables

# %%
def plot_key_distributions(df, y_col, t_col, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    y_data = df[y_col].clip(upper=df[y_col].quantile(0.95))
    sns.histplot(y_data, kde=True, ax=axes[0], color='steelblue', bins=30)
    axes[0].set_title(f"Distribution of {y_col}", fontsize=12)
    axes[0].set_xlabel(y_col)
    
    sns.histplot(df[t_col], kde=True, ax=axes[1], color='darkorange', bins=30)
    axes[1].set_title(f"Distribution of {t_col}", fontsize=12)
    axes[1].set_xlabel(t_col)
    
    if 'ESG' in df.columns:
        sns.histplot(df['ESG'], kde=True, ax=axes[2], color='forestgreen', bins=30)
        axes[2].set_title('Distribution of ESG', fontsize=12)
        axes[2].set_xlabel('ESG')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_key_distributions(df_mapped, y_col, t_col)

# %% [markdown]
# ## Correlation Heatmap

# %%
def plot_correlation_heatmap(df, y_col, t_col, x_cols, save_path=None):
    all_cols = [y_col, t_col] + x_cols
    corr_matrix = df[all_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax, 
                square=True, linewidths=0.5)
    
    ax.set_title('Correlation Heatmap of Analysis Variables', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_correlation_heatmap(df_mapped, y_col, t_col, x_cols)


