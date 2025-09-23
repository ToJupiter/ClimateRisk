#!/usr/bin/env python3
"""
Complete Causal Machine Learning Analysis
Following exact specifications for production-grade causal inference analysis
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy import stats
import statsmodels.api as sm

# Causal ML libraries
from econml.dml import LinearDML
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')
np.random.seed(42)

class CausalAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.results = {}
        
    def load_and_examine_data(self):
        """Load dataset and examine structure"""
        print("=" * 60)
        print("STEP 1: DATA LOADING & EXAMINATION")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Original dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check data types and missing values
        print("\nMissing values per column:")
        missing_counts = self.df.isnull().sum()
        print(missing_counts[missing_counts > 0])
        
        return self.df
    
    def filter_and_map_variables(self):
        """Filter missing values and map variables to exact specification"""
        print("\n" + "=" * 60)
        print("STEP 1C & 2: DATA FILTERING & VARIABLE MAPPING")
        print("=" * 60)
        
        # Define required variables for analysis
        analysis_vars = {
            # Outcome variable
            'Tobins_Q': 'FV',
            # Treatment variable  
            'Climate_Risk_TFIDF': 'ClimateRiskScore',
            # Moderator
            'ESG': 'ESG',
            # Confounders - map to available columns
            'SIZE': 'SIZE',
            'LEV': 'LEV', 
            'STDebt_TA': 'lctat',
            'STDebt_TL': 'lctlt',
            'IntExp_Sales': 'chlct',
            # 'CashFlow': 'CF',
            'Cash_TA_lag': 'cheat',
            'Inventory_Sales': 'Inventory',
            # 'Tangible_Asset_Ratio': 'Fixed',
            'Growth': 'Growth',
            'Net_Income_After_Tax': 'NI',
            'GDP_Growth': 'GDPgrowth',
            'Inf': 'INF',
            # Additional variables
            'Country': 'Country of Exchange',
            'Year': 'year',
            'GICS': 'GICS Industry Name'
        }
        
        # Create mapped dataset
        mapped_df = pd.DataFrame()
        for new_name, old_name in analysis_vars.items():
            if old_name in self.df.columns:
                mapped_df[new_name] = self.df[old_name]
            else:
                print(f"Warning: Column {old_name} not found for {new_name}")
        
        print(f"Mapped dataset shape: {mapped_df.shape}")
        
        # Remove rows with ANY missing values in analysis variables
        print(f"Missing values before filtering:")
        print(mapped_df.isnull().sum().sum())
        
        initial_rows = len(mapped_df)
        mapped_df = mapped_df.dropna()
        final_rows = len(mapped_df)
        
        print(f"Rows removed due to missing values: {initial_rows - final_rows}")
        print(f"Final dataset shape: {mapped_df.shape}")
        
        self.df = mapped_df
        return self.df
    
    def prepare_case_data(self, include_confounders=True, include_moderator=False):
        """Prepare data for specific analysis case"""
        # Define confounders (X)
        confounder_cols = [
            'SIZE', 'LEV', 'STDebt_TA', 'STDebt_TL', 'IntExp_Sales',
            'Cash_TA_lag',
            'Inventory_Sales', 'Growth',
            'Net_Income_After_Tax', 'GDP_Growth', 'Inf'
        ]
        
        # Treatment variable
        T = self.df['Climate_Risk_TFIDF'].values.reshape(-1, 1)
        # Outcome variable
        Y = self.df['Tobins_Q'].values
        
        # Confounders
        if include_confounders:
            X = self.df[confounder_cols].values
        else:
            X = None
            
        # Moderator
        if include_moderator:
            W = self.df['ESG'].values.reshape(-1, 1)
        else:
            W = None
            
        return Y, T, X, W, confounder_cols
    
    def run_case_analysis(self, case_num, include_confounders=True, include_moderator=False):
        """Run specific causal analysis case using LinearDML"""
        print(f"\n" + "=" * 60)
        print(f"CASE {case_num}: {'WITH' if include_confounders else 'WITHOUT'} CONFOUNDERS, "
              f"{'WITH' if include_moderator else 'WITHOUT'} MODERATOR")
        print("=" * 60)
        
        Y, T, X, W, confounder_cols = self.prepare_case_data(include_confounders, include_moderator)
        
        if case_num in [1, 3]:
            model_treatment = RandomForestRegressor(n_estimators=100, random_state=42)
            model_outcome = RandomForestRegressor(n_estimators=100, random_state=42)
        else: 
            model_treatment = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                reg_alpha=0,
                reg_lambda=1            
            )
            model_outcome = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                reg_alpha=0.1,
                reg_lambda=0.5            
                )
        
        dml = LinearDML(
            model_y=model_outcome,
            model_t=model_treatment,
            cv=5,
            random_state=42
        )
        
        if include_moderator:
            dml.fit(Y, T, X=X, W=W)
        else:
            dml.fit(Y, T, X=X)
        
        # Get treatment effect
        if include_confounders:
            effect = dml.effect(X)
        else:
            effect = dml.effect()
            
        effect_mean = np.mean(effect)
        
        # Get confidence intervals
        if include_confounders:
            effect_interval = dml.effect_interval(X, alpha=0.05)
        else:
            effect_interval = dml.effect_interval(alpha=0.05)
            
        ci_lower = np.mean(effect_interval[0])
        ci_upper = np.mean(effect_interval[1])
        
        if include_confounders:
            inference_result = dml.effect_inference(X)
        else:
            inference_result = dml.effect_inference()
            
        try:
            treatment_p = inference_result.pvalue().mean() 
        except:
            effect_se = (ci_upper - ci_lower) / (2 * 1.96)
            treatment_t = effect_mean / effect_se if effect_se > 0 else 0
            treatment_p = 2 * (1 - stats.norm.cdf(np.abs(treatment_t)))
        
        # Store results
        case_key = f"case_{case_num}"
        self.results[case_key] = {
            'model': dml,
            'effect_mean': effect_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': treatment_p
        }
        
        print(f"Case {case_num} Results:")
        print(f"Average Treatment Effect (ATE): {effect_mean:.6f}")
        print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"p-value: {treatment_p:.6f}")
        
        return self.results[case_key]
    
    def run_all_cases(self):
        """Run all four causal analysis cases"""
        print("Running all four causal analysis cases...")
        
        self.run_case_analysis(1, include_confounders=False, include_moderator=False)        
        self.run_case_analysis(2, include_confounders=True, include_moderator=False)        
        self.run_case_analysis(3, include_confounders=False, include_moderator=True)        
        self.run_case_analysis(4, include_confounders=True, include_moderator=True)
        
        return self.results
    
    def plot_distribution_plots(self):
        """Plot histograms + KDE for Tobins_Q, Climate_Risk_TFIDF, ESG (3 columns horizontally, 3 colors)"""
        vars_to_plot = ['Tobins_Q', 'Climate_Risk_TFIDF', 'ESG']
        titles = ['Firm Value (Tobin\'s Q)', 'Climate Risk (TFIDF)', 'ESG Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bins_list = [60, 30, 30]  # More bins for Tobins_Q for finer resolution

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (var, color) in enumerate(zip(vars_to_plot, colors)):
            data = self.df[var]
            if var == 'Tobins_Q':
                # Clip extreme values for better visualization
                data = data.clip(upper=15)
                axes[i].set_xlim(0, 14)
            sns.histplot(data, kde=True, ax=axes[i], bins=bins_list[i], color=color)
            axes[i].set_title(f'{titles[i]} - Distribution')
            axes[i].set_xlabel(var)

        plt.tight_layout()
        plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: distribution_plots.png")

    def plot_cate_by_esg(self):
        """Plot Conditional Average Treatment Effect (CATE) by ESG score"""
        case4 = self.results.get('case_4')
        if not case4:
            print("Case 4 model not found. Run case 4 first.")
            return

        Y, T, X, W, _ = self.prepare_case_data(include_confounders=True, include_moderator=True)
        dml = case4['model']
        cate = dml.effect(X, T0=0, T1=1)  # or just dml.effect(X) if binary/default treatment

        # Bin ESG scores for smoother plot
        esg_series = self.df['ESG']
        bins = pd.qcut(esg_series, q=10, duplicates='drop')
        cate_series = pd.Series(cate, index=esg_series.index)

        grouped = cate_series.groupby(bins).agg(['mean', 'count', 'std'])
        grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['std'] / np.sqrt(grouped['count'])

        plt.figure(figsize=(10, 6))
        plt.plot(grouped.index.astype(str).str.replace(', ', ' to ').str.strip('()'), grouped['mean'], 
                 marker='o', linestyle='-', label='CATE')
        plt.fill_between(range(len(grouped)), grouped['ci_lower'], grouped['ci_upper'], alpha=0.3, label='95% CI')
        plt.title('CATE of Climate Risk on Firm Value by ESG Score')
        plt.xlabel('ESG Score Bins (Quantiles)')
        plt.ylabel('Estimated Treatment Effect')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cate_by_esg.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: cate_by_esg.png")

    def plot_shap_feature_importance(self):
        """Plot SHAP feature importance for outcome model in Case 4"""
        case4 = self.results.get('case_4')
        if not case4:
            print("Case 4 model not found. Run case 4 first.")
            return

        _, _, X, _, confounder_cols = self.prepare_case_data(include_confounders=True, include_moderator=True)
        model_y = case4['model'].model_y


        # Create SHAP explainer
        explainer = shap.Explainer(model_y, X, feature_names=confounder_cols)
        shap_values = explainer(X)

        # Plot summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=confounder_cols, show=False)
        plt.title("SHAP Feature Importance for Predicting Tobin's Q")
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: shap_feature_importance.png")

def causal_main(path_to_combined_csv: str):
    """Main execution function"""
    print("Starting Complete Causal Machine Learning Analysis")
    print("Following exact specifications for production-grade analysis")
    
    # Initialize analysis with temporary file
    analysis = CausalAnalysis(path_to_combined_csv)
    
    # Execute all steps
    try:
        # Step 1: Load and validate data
        analysis.load_and_examine_data()
        analysis.filter_and_map_variables()
        
        # Run all four cases
        results = analysis.run_all_cases()

        analysis.plot_distribution_plots()
        analysis.plot_cate_by_esg()
        # Should add here
        # analysis.plot_shap_feature_importance()
        
        # Print results
        for case_key, result in results.items():
            print(f"\n{case_key.upper()} RESULTS:")
            print(f"  Average Treatment Effect (ATE): {result['effect_mean']:.6f}")
            print(f"  95% Confidence Interval: [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
            print(f"  p-value: {result['p_value']:.6f}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    causal_main()