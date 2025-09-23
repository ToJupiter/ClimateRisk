import polars as pl

# --- 1. Configuration ---
input_filename = "Combined_Company_Data_2022_2024.csv"
output_filename = "Combined_Company_Data_2022_2024_Final1.csv"

# --- 2. Load the dataset ---
print(f"Loading the completed dataset: '{input_filename}'...")
try:
    df = pl.read_csv(input_filename)
    # The first column is an old index, we can drop it.
    if "" in df.columns:
        df = df.drop("")
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run the previous script first to generate the input file.")
    exit()

# --- 3. Filter for companies with data for all 3 years ---
print("\nFiltering for companies with complete data for 2022, 2023, and 2024...")

# Get the count of years for each company
company_counts = df.group_by("Identifier").agg(pl.count().alias("year_count"))

# Filter to find identifiers that have exactly 3 years of data
complete_companies = company_counts.filter(pl.col("year_count") == 3)

# Filter the original dataframe to keep only the complete companies
df_filtered = df.filter(pl.col("Identifier").is_in(complete_companies["Identifier"]))

original_companies = df["Identifier"].n_unique()
final_companies = df_filtered["Identifier"].n_unique()

print(f"Identified {final_companies} companies with complete data (out of {original_companies}).")
print(f"Filtered dataset shape: {df_filtered.shape}")


# --- 4. Add a new ID column ---
print("\nAdding a new sequential ID for each company...")

# Get a sorted list of unique identifiers
unique_identifiers = df_filtered.select("Identifier").unique().sort("Identifier")

# Create a new DataFrame with identifiers and a new sequential ID
id_map = unique_identifiers.with_columns(
    pl.int_range(1, pl.count() + 1).alias("ID")
)

# Join the new ID back to the filtered dataframe
df_final = df_filtered.join(id_map, on="Identifier")

# Reorder columns to have 'ID' at the beginning
cols = [col for col in df_final.columns if col != "ID"]
df_final = df_final.select(cols)


# --- 5. Save the final result ---
df_final.write_csv(output_filename)

print(f"\nSuccessfully created the final, cleaned dataset: '{output_filename}'")
print("Final data sample:")
print(df_final.head())