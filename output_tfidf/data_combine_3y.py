import polars as pl

old_company_data = pl.read_csv("Combined_Company_Data.csv")
new_3y_data = pl.read_csv("ESG.csv")

result = new_3y_data.join(
    old_company_data.select(["year", "Identifier", "ClimateRiskScore"]),
    on=["year", "Identifier"],
    how='left'
)

# print(result.head(7))
with open("Combined_Company_Data_2022_2024.csv", "w", encoding='utf-8') as dumpo:
    result.write_csv(dumpo)