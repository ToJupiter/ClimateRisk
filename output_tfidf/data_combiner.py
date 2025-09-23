import pandas as pd
import re
import os

def combine_climate_risk_data():
    tfidf_df = pd.read_csv('tfidf_scores.csv')
    company_results_df = pd.read_csv('company_identification_results.csv')
    company_dataset_df = pd.read_csv('Company_Dataset_Final.csv')
    
    tfidf_clean = tfidf_df[['doc_id', 'ClimateRiskScore']].copy()
    
    tfidf_clean['year'] = tfidf_clean['doc_id'].str.extract(r'(2018|2019|2020|2021|2022|2023|2024)')
    tfidf_clean['path_to_file'] = tfidf_clean['doc_id']
    
    tfidf_clean['folder_match'] = tfidf_clean['doc_id'].apply(
        lambda x: '/'.join(x.split('/')[:-1]) if '/' in x else x.split()[0]
    )
    
    company_results_clean = company_results_df[
        ['folder_path', 'company_name', 'company_code']
    ].copy()
    
    folder_to_company = {}
    for _, row in company_results_clean.iterrows():
        folder_to_company[row['folder_path']] = {
            'company_name': row['company_name'],
            'company_code': row['company_code']
        }
    
    def map_company_info(folder_match):
        for folder_path, company_info in folder_to_company.items():
            if folder_match in folder_path or folder_path in folder_match:
                return pd.Series([company_info['company_name'], company_info['company_code']])
        return pd.Series([None, None])
    
    tfidf_clean[['company_name', 'company_code']] = tfidf_clean['folder_match'].apply(map_company_info)
    
    intermediate_df = tfidf_clean[
        ['path_to_file', 'year', 'company_name', 'company_code', 'ClimateRiskScore']
    ].dropna()
    
    intermediate_df['year'] = pd.to_numeric(intermediate_df['year'], errors='coerce')
    
    final_df = intermediate_df.merge(
        company_dataset_df,
        left_on=['company_code', 'year'],
        right_on=['Identifier', 'Year'],
        how='left'
    )
    
    final_df.to_csv('combined_climate_risk_data.csv', index=False)
    
    print(f"Combined dataset created with {len(final_df)} rows")
    print(f"Columns: {list(final_df.columns)}")
    
    return final_df

if __name__ == "__main__":
    result = combine_climate_risk_data()
