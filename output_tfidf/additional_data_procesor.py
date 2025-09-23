import polars as pl
import re

# --- 1. Load the datasets ---
print("Loading datasets...")
try:
    # Load the main company data
    main_df = pl.read_csv("Combined_Company_Data_2022_2024.csv")
    
    # Load the additional TF-IDF scores
    additional_tfidf_df = pl.read_csv("additional_tfidf_scores.csv")
    
    print("Datasets loaded successfully.")
    print(f"Main data shape: {main_df.shape}")
    print(f"Additional scores shape: {additional_tfidf_df.shape}")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both CSV files are in the correct directory.")
    exit()

# --- 2. Process the additional TF-IDF scores data ---
print("\nProcessing additional TF-IDF scores...")

# Keep only the necessary columns
additional_df_processed = additional_tfidf_df.select(
    pl.col("doc_id"),
    pl.col("ClimateRiskScore").alias("New_ClimateRiskScore") # Rename to avoid conflict during join
)

# Use regex to extract 'Identifier' and 'year' from 'doc_id'
# The regex captures the part before the year (Identifier) and the 4-digit year at the end.
# It handles cases with or without a separator like '_'.
additional_df_processed = additional_df_processed.with_columns(
    pl.col("doc_id").str.extract(r"^(.*?)_?(\d{4})$", 1).alias("Identifier"),
    pl.col("doc_id").str.extract(r"^(.*?)_?(\d{4})$", 2).cast(pl.Int64).alias("year")
)

# Drop the original doc_id as it's no longer needed
additional_df_processed = additional_df_processed.drop("doc_id")

print("Finished processing additional scores. Extracted 'Identifier' and 'year'.")
print("Sample of processed additional data:")
print(additional_df_processed.head())


# --- 3. Merge the scores into the main dataframe ---
print("\nMerging additional scores into the main dataset...")

# Perform a left join to bring the new scores into the main dataframe
# The join keys are the company 'Identifier' and the 'year'
merged_df = main_df.join(
    additional_df_processed, 
    on=["Identifier", "year"], 
    how="left"
)

# Use coalesce to fill missing 'ClimateRiskScore' values with 'New_ClimateRiskScore'
# coalesce takes the first non-null value from the list of columns.
final_df = merged_df.with_columns(
    pl.coalesce(pl.col("ClimateRiskScore"), pl.col("New_ClimateRiskScore")).alias("ClimateRiskScore")
).drop("New_ClimateRiskScore") # Drop the temporary new score column

# Calculate how many missing values were filled
original_missing_count = main_df['ClimateRiskScore'].is_null().sum()
new_missing_count = final_df['ClimateRiskScore'].is_null().sum()
filled_count = original_missing_count - new_missing_count

print(f"Filled {filled_count} missing ClimateRiskScore values.")


# --- 4. Clean the final dataset ---
print(f"\nTotal rows before final cleaning: {final_df.shape[0]}")
print(f"Rows with missing ClimateRiskScore before cleaning: {new_missing_count}")

# Filter out any rows that still have a missing ClimateRiskScore
cleaned_df = final_df.filter(pl.col("ClimateRiskScore").is_not_null())

print(f"Total rows after final cleaning: {cleaned_df.shape[0]}")
print(f"Rows with missing ClimateRiskScore after cleaning: {cleaned_df['ClimateRiskScore'].is_null().sum()}")


# --- 5. Save the result ---
output_filename = "Combined_Company_Data_2022_2024.csv"
cleaned_df.write_csv(output_filename)

print(f"\nSuccessfully created the complete and cleaned dataset: '{output_filename}'")
print("Final data sample:")
print(cleaned_df.head())