#!/usr/bin/env python3
"""
Complete Causal Machine Learning Analysis
Following exact specifications for production-grade causal inference analysis
"""

import pandas as pd
import numpy as np
import warnings
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
            
        # Extract p-value for the treatment effect
        try:
            # Get p-value directly from inference result
            treatment_p = inference_result.pvalue().mean() # Average p-value across samples
        except:
            # Fallback calculation using mean effect and CI
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

def main():
    """Main execution function"""
    print("Starting Complete Causal Machine Learning Analysis")
    print("Following exact specifications for production-grade analysis")
    
    # Initialize analysis with temporary file
    analysis = CausalAnalysis("/mnt/e/NEUConference/ClimateRisk/output_tfidf/Combined_Company_Data_2022_2024_Final2.csv")
    
    # Execute all steps
    try:
        # Step 1: Load and validate data
        analysis.load_and_examine_data()
        analysis.filter_and_map_variables()
        
        # Run all four cases
        results = analysis.run_all_cases()
        
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
    main()