import pandas as pd

# List of Excel files to merge
excel_files = ["english.csv", "hindi.csv", "kannad.csv", "roman_hind.csv","tamil.csv"]

# List to hold dataframes of each Excel file
dfs = []

# Read each Excel file and store it as a dataframe in dfs list
for file in excel_files:
    dfs.append(pd.read_csv(file))

# Merge all dataframes in the list dfs
merged_df = pd.concat(dfs)

# Write the merged dataframe to a new Excel file
merged_df.to_excel("merged_files.xlsx", index=False)

