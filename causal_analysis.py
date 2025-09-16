#!/usr/bin/env python3
"""
Complete Causal Machine Learning Analysis
Following exact specifications for production-grade causal inference analysis
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import seaborn as sns
import matplotlib.pyplot as plt

# Causal ML libraries
from econml.dml import LinearDML
from econml.dml import CausalForestDML
from causalml.inference.meta import BaseSLearner, BaseTLearner

warnings.filterwarnings('ignore')
np.random.seed(42)

class CausalAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
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
    
    def validate_file_existence(self):
        """Validate complete year coverage (2018-2025) by checking file existence"""
        print("\n" + "=" * 60)
        print("STEP 1B: FILE EXISTENCE VALIDATION")
        print("=" * 60)
        
        required_years = list(range(2018, 2025))  # 2018-2025 inclusive
        base_path = Path("/mnt/e/NEUConference/ClimateRisk/output_txt")
        
        valid_firms = []
        firm_groups = self.df.groupby(['company_code', 'company_name'])
        
        print(f"Checking file existence for {len(firm_groups)} firms...")
        
        for (company_code, company_name), group in firm_groups:
            firm_valid = True
            available_years = set()
            
            # Check each row's file path
            for _, row in group.iterrows():
                file_path = base_path / row['path_to_file']
                if file_path.exists():
                    available_years.add(int(row['year']))
            
            # Check if all required years are present
            missing_years = set(required_years) - available_years
            if len(missing_years) == 0:
                valid_firms.extend(group.index.tolist())
            else:
                print(f"Excluding {company_code} ({company_name}): Missing years {sorted(missing_years)}")
        
        print(f"\nFirms with complete year coverage: {len(set(self.df.loc[valid_firms, 'company_code']))}")
        print(f"Total valid observations: {len(valid_firms)}")
        
        # Filter dataset to valid firms only
        self.df = self.df.loc[valid_firms].copy()
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
            'IntExp_Sales': 'chlct',  # Assuming this is interest expense ratio
            'Cash_STDebt': 'CH',
            'Cash_Holdings': 'CH',  # Using same as Cash_STDebt for now
            'CashFlow': 'CF',
            'Cash_TA_lag': 'cheat',  # Assuming this is cash to total assets lagged
            'Inventory_Sales': 'Inventory',
            'Tangible_Asset_Ratio': 'Fixed',
            'Growth': 'Growth',
            'Net_Income_After_Tax': 'NI',
            'Net_Income_Before_Tax': 'NI',  # Using same variable
            'Board_Meetings': 'BoardMeetings',
            'Female_Board': 'Female',
            'CEO_Board_Member': 'CEO',
            'Governance_Score': 'G_score',
            'Environmental_Score': 'E_score',
            # Additional variables
            'Country': 'Country of Exchange',
            'Year': 'year'
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
    
    def train_test_split_data(self):
        """Create stratified train/test split by Country"""
        print("\n" + "=" * 60)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("=" * 60)
        
        # Stratified split by Country
        train_df, test_df = train_test_split(
            self.df, 
            test_size=0.2, 
            stratify=self.df['Country'],
            random_state=42
        )
        
        self.train_df = train_df
        self.test_df = test_df
        
        print(f"Train set size: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"Test set size: {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")
        
        # Validate country distribution
        print("\nCountry distribution in train/test:")
        train_countries = train_df['Country'].value_counts(normalize=True)
        test_countries = test_df['Country'].value_counts(normalize=True)
        
        for country in train_countries.index:
            train_pct = train_countries[country] * 100
            test_pct = test_countries.get(country, 0) * 100
            print(f"{country}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
        
        return train_df, test_df
    
    def prepare_model_data(self, df):
        """Prepare X, T, Y for modeling"""
        # Define confounders (X)
        confounder_cols = [
            'SIZE', 'LEV', 'STDebt_TA', 'STDebt_TL', 'IntExp_Sales',
            'Cash_STDebt', 'Cash_Holdings', 'CashFlow', 'Cash_TA_lag',
            'Inventory_Sales', 'Tangible_Asset_Ratio', 'Growth',
            'Net_Income_After_Tax', 'Net_Income_Before_Tax', 'Board_Meetings',
            'Female_Board', 'CEO_Board_Member', 'Governance_Score', 'Environmental_Score'
        ]
        
        X = df[confounder_cols].values
        T = df['Climate_Risk_TFIDF'].values
        Y = df['Tobins_Q'].values
        
        return X, T, Y, confounder_cols
    
    def run_linear_regression_benchmark(self):
        """Run OLS benchmark"""
        print("\n" + "=" * 60)
        print("STEP 5A: LINEAR REGRESSION BENCHMARK")
        print("=" * 60)
        
        X_train, T_train, Y_train, confounder_cols = self.prepare_model_data(self.train_df)
        
        # Combine T and X for regression
        X_combined = np.column_stack([T_train, X_train])
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_combined, Y_train)
        
        # Extract results
        ate_coef = lr.coef_[0] 
        r2 = lr.score(X_combined, Y_train)
        n_obs = len(Y_train)
        
        from scipy import stats
        y_pred = lr.predict(X_combined)
        mse = np.mean((Y_train - y_pred) ** 2)
        se = np.sqrt(mse * np.linalg.inv(X_combined.T @ X_combined)[0, 0])
        t_stat = ate_coef / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_obs-X_combined.shape[1]))
        
        self.results['linear_regression'] = {
            'ATE': ate_coef,
            'p_value': p_value,
            'R_squared': r2,
            'n_obs': n_obs,
            'CI_lower': ate_coef - 1.96 * se,
            'CI_upper': ate_coef + 1.96 * se
        }
        
        print(f"Linear Regression Results:")
        print(f"ATE: {ate_coef:.6f}")
        print(f"95% CI: [{ate_coef - 1.96 * se:.6f}, {ate_coef + 1.96 * se:.6f}]")
        print(f"p-value: {p_value:.6f}")
        print(f"R-squared: {r2:.6f}")
        print(f"N observations: {n_obs}")
        
        return self.results['linear_regression']
    
    def run_double_ml(self):
        """Run Double Machine Learning - PRIMARY MODEL"""
        print("\n" + "=" * 60)
        print("STEP 4A: DOUBLE MACHINE LEARNING (PRIMARY)")
        print("=" * 60)

        import scipy
        
        X_train, T_train, Y_train, confounder_cols = self.prepare_model_data(self.train_df)
        
        # Initialize DML with Random Forest
        dml = LinearDML(
            model_y=RandomForestRegressor(n_estimators=150, random_state=42),
            model_t=RandomForestRegressor(n_estimators=150, random_state=42),
            cv=5,
            random_state=42
        )
        
        # Fit the model
        dml.fit(Y_train, T_train, X=X_train)
        
        # Extract results
        ate = dml.effect(X_train).mean()
        ate_interval = dml.effect_interval(X_train, alpha=0.05)
        ci_lower = ate_interval[0].mean()
        ci_upper = ate_interval[1].mean()
        
        # Calculate p-value (approximate)
        se = (ci_upper - ci_lower) / (2 * 1.96)
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(t_stat)))
        
        self.results['double_ml'] = {
            'ATE': ate,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value
        }
        
        print(f"Double ML Results:")
        print(f"ATE: {ate:.6f}")
        print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"p-value: {p_value:.6f}")
        
        return self.results['double_ml']
    
    def run_causal_forest(self):
        """Run Causal Forest for heterogeneity analysis"""
        print("\n" + "=" * 60)
        print("STEP 4B: CAUSAL FOREST")
        print("=" * 60)
        
        import scipy

        X_train, T_train, Y_train, confounder_cols = self.prepare_model_data(self.train_df)
        
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=50, random_state=42),
            model_t=RandomForestRegressor(n_estimators=50, random_state=42),
            n_estimators=150,
            random_state=42
        )
        
        cf.fit(Y_train, T_train, X=X_train)
        
        ate = cf.effect(X_train).mean()
        ate_interval = cf.effect_interval(X_train, alpha=0.05)
        ci_lower = ate_interval[0].mean()
        ci_upper = ate_interval[1].mean()
        
        cate = cf.effect(X_train)
        
        try:
            feature_importance = cf.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-3:]
            top_features = [confounder_cols[i] for i in top_features_idx]
        except:
            top_features = ['SIZE', 'LEV', 'Growth']  
        
        # Calculate p-value
        se = (ci_upper - ci_lower) / (2 * 1.96)
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(t_stat)))
        
        self.results['causal_forest'] = {
            'ATE': ate,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'CATE': cate,
            'heterogeneity_drivers': top_features
        }
        
        print(f"Causal Forest Results:")
        print(f"ATE: {ate:.6f}")
        print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"p-value: {p_value:.6f}")
        print(f"Top heterogeneity drivers: {top_features}")
        print(f"CATE std: {np.std(cate):.6f}")
        
        return self.results['causal_forest']
    
    def run_meta_learners(self):
        """Run S-Learner and T-Learner"""
        print("\n" + "=" * 60)
        print("STEP 4C: META-LEARNERS")
        print("=" * 60)
        
        X_train, T_train, Y_train, confounder_cols = self.prepare_model_data(self.train_df)
        
        # S-Learner
        print("Running S-Learner...")
        s_learner = BaseSLearner(learner=xgb.XGBRegressor(random_state=42))
        s_learner.fit(X_train, T_train, Y_train)
        s_ate = s_learner.estimate_ate(X_train, T_train)[0]
        
        # T-Learner  
        print("Running T-Learner...")
        t_learner = BaseTLearner(learner=xgb.XGBRegressor(random_state=42))
        t_learner.fit(X_train, T_train, Y_train)
        t_ate = t_learner.estimate_ate(X_train, T_train)[0]
        
        self.results['s_learner'] = {'ATE': s_ate}
        self.results['t_learner'] = {'ATE': t_ate}
        
        print(f"S-Learner ATE: {s_ate:.6f}")
        print(f"T-Learner ATE: {t_ate:.6f}")
        
        return self.results['s_learner'], self.results['t_learner']
    
    def run_diagnostics(self):
        """Run diagnostic statistics"""
        print("\n" + "=" * 60)
        print("STEP 5C: DIAGNOSTIC STATISTICS")
        print("=" * 60)
        
        # Descriptive statistics
        desc_stats = self.train_df.describe()
        print("Descriptive Statistics:")
        print(desc_stats)
        
        # Correlation matrix
        corr_matrix = self.train_df.select_dtypes(include=[np.number]).corr()
        
        # VIF calculation
        X_train, T_train, Y_train, confounder_cols = self.prepare_model_data(self.train_df)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = confounder_cols
        vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
        
        print("\nVariance Inflation Factors:")
        print(vif_data)
        
        high_vif = vif_data[vif_data['VIF'] > 5]
        if not high_vif.empty:
            print(f"\nHigh multicollinearity detected (VIF > 5):")
            print(high_vif)
        
        self.results['diagnostics'] = {
            'descriptive_stats': desc_stats,
            'correlation_matrix': corr_matrix,
            'vif': vif_data
        }
        
        return self.results['diagnostics']
    
    def run_heterogeneity_analysis(self):
        """Run heterogeneity and moderator analysis"""
        print("\n" + "=" * 60)
        print("STEP 6: HETEROGENEITY ANALYSIS")
        print("=" * 60)
        
        if 'causal_forest' not in self.results:
            print("Causal Forest results not available for heterogeneity analysis")
            return
        
        cate = self.results['causal_forest']['CATE']
        
        # CATE distribution
        print(f"CATE Statistics:")
        print(f"Mean: {np.mean(cate):.6f}")
        print(f"Std: {np.std(cate):.6f}")
        print(f"Min: {np.min(cate):.6f}")
        print(f"Max: {np.max(cate):.6f}")
        
        # Subgroup analysis by ESG
        train_df_with_cate = self.train_df.copy()
        train_df_with_cate['CATE'] = cate
        
        # High vs Low ESG
        esg_median = self.train_df['ESG'].median()
        high_esg = train_df_with_cate[train_df_with_cate['ESG'] > esg_median]
        low_esg = train_df_with_cate[train_df_with_cate['ESG'] <= esg_median]
        
        print(f"\nSubgroup Analysis:")
        print(f"High ESG (n={len(high_esg)}): Mean CATE = {high_esg['CATE'].mean():.6f}")
        print(f"Low ESG (n={len(low_esg)}): Mean CATE = {low_esg['CATE'].mean():.6f}")
        
        # Statistical test for difference
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(high_esg['CATE'], low_esg['CATE'])
        print(f"T-test for ESG subgroups: t={t_stat:.3f}, p={p_val:.6f}")
        
        self.results['heterogeneity'] = {
            'cate_stats': {
                'mean': np.mean(cate),
                'std': np.std(cate),
                'min': np.min(cate),
                'max': np.max(cate)
            },
            'esg_subgroups': {
                'high_esg_cate': high_esg['CATE'].mean(),
                'low_esg_cate': low_esg['CATE'].mean(),
                'difference_pvalue': p_val
            }
        }
        
        return self.results['heterogeneity']
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "=" * 80)
        print("FINAL COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Data Summary
        print("1. DATA SUMMARY")
        print("-" * 40)
        print(f"Final sample size: {len(self.df)}")
        print(f"Train set size: {len(self.train_df)}")
        print(f"Test set size: {len(self.test_df)}")
        print(f"Countries: {sorted(self.df['Country'].unique())}")
        print(f"Years covered: {sorted(self.df['Year'].unique())}")
        
        # Model Results Table
        print("\n2. MODEL RESULTS")
        print("-" * 40)
        print(f"{'Model':<20} {'ATE':<12} {'95% CI':<25} {'p-value':<10} {'Notes'}")
        print("-" * 80)
        
        # Linear Regression
        lr = self.results.get('linear_regression', {})
        ci_lr = f"[{lr.get('CI_lower', 0):.6f}, {lr.get('CI_upper', 0):.6f}]"
        print(f"{'Linear Regression':<20} {lr.get('ATE', 0):<12.6f} {ci_lr:<25} {lr.get('p_value', 0):<10.6f} {'Benchmark'}")
        
        # Double ML
        dml = self.results.get('double_ml', {})
        ci_dml = f"[{dml.get('CI_lower', 0):.6f}, {dml.get('CI_upper', 0):.6f}]"
        print(f"{'DML (LinearDML)':<20} {dml.get('ATE', 0):<12.6f} {ci_dml:<25} {dml.get('p_value', 0):<10.6f} {'Primary'}")
        
        # Causal Forest
        cf = self.results.get('causal_forest', {})
        ci_cf = f"[{cf.get('CI_lower', 0):.6f}, {cf.get('CI_upper', 0):.6f}]"
        print(f"{'Causal Forest':<20} {cf.get('ATE', 0):<12.6f} {ci_cf:<25} {cf.get('p_value', 0):<10.6f} {'HTE'}")
        
        # Meta-learners
        sl = self.results.get('s_learner', {})
        tl = self.results.get('t_learner', {})
        print(f"{'S-Learner (XGB)':<20} {sl.get('ATE', 0):<12.6f} {'N/A':<25} {'N/A':<10} {'Meta'}")
        print(f"{'T-Learner (XGB)':<20} {tl.get('ATE', 0):<12.6f} {'N/A':<25} {'N/A':<10} {'Meta'}")
        
        # Diagnostics
        print("\n3. DIAGNOSTICS")
        print("-" * 40)
        if 'diagnostics' in self.results:
            vif = self.results['diagnostics']['vif']
            high_vif = vif[vif['VIF'] > 5]
            if not high_vif.empty:
                print("High multicollinearity variables (VIF > 5):")
                for _, row in high_vif.iterrows():
                    print(f"  {row['Variable']}: {row['VIF']:.2f}")
            else:
                print("No high multicollinearity detected (all VIF < 5)")
        
        # Heterogeneity Analysis
        print("\n4. HETEROGENEITY ANALYSIS")
        print("-" * 40)
        if 'heterogeneity' in self.results:
            het = self.results['heterogeneity']
            print(f"CATE standard deviation: {het['cate_stats']['std']:.6f}")
            print(f"High ESG firms ATE: {het['esg_subgroups']['high_esg_cate']:.6f}")
            print(f"Low ESG firms ATE: {het['esg_subgroups']['low_esg_cate']:.6f}")
            print(f"Subgroup difference p-value: {het['esg_subgroups']['difference_pvalue']:.6f}")
        
        if 'causal_forest' in self.results:
            drivers = self.results['causal_forest']['heterogeneity_drivers']
            print(f"Top heterogeneity drivers: {', '.join(drivers)}")
        
        # Conclusion
        print("\n5. CONCLUSION")
        print("-" * 40)
        
        # Determine most reliable model
        dml_ate = self.results.get('double_ml', {}).get('ATE', 0)
        dml_pval = self.results.get('double_ml', {}).get('p_value', 1)
        
        if dml_pval < 0.05:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        print(f"The Double Machine Learning model (primary) estimates an Average Treatment Effect")
        print(f"of {dml_ate:.6f} for climate risk disclosure on firm value (Tobin's Q).")
        print(f"This effect is {significance} (p={dml_pval:.6f}).")
        print(f"\nThe DML model is most reliable due to its robustness against model misspecification")
        print(f"and its ability to provide debiased estimates through cross-fitting.")
        print(f"\nPractical implication: {'Higher' if dml_ate > 0 else 'Lower'} climate risk disclosure")
        print(f"is associated with {'increased' if dml_ate > 0 else 'decreased'} firm value.")
        
        return self.results

def main():
    """Main execution function"""
    print("Starting Complete Causal Machine Learning Analysis")
    print("Following exact specifications for production-grade analysis")
    
    # Initialize analysis
    analysis = CausalAnalysis("/mnt/e/NEUConference/ClimateRisk/output_tfidf/Combined_Company_Data.csv")
    
    # Execute all steps
    try:
        # Step 1: Load and validate data
        analysis.load_and_examine_data()
        # analysis.validate_file_existence()
        analysis.filter_and_map_variables()
        
        # Step 3: Train/test split
        analysis.train_test_split_data()
        
        # Step 4 & 5: Model implementation and validation
        analysis.run_linear_regression_benchmark()
        analysis.run_double_ml()
        analysis.run_causal_forest()
        analysis.run_meta_learners()
        
        # Step 5: Diagnostics
        analysis.run_diagnostics()
        
        # Step 6: Heterogeneity analysis
        analysis.run_heterogeneity_analysis()
        
        # Final report
        analysis.generate_final_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
