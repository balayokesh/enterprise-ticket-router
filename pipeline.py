import pandas as pd

try:
    df = pd.read_csv("data.csv")
    print(f"Successfully loaded 'data.csv' with {len(df)} rows and {len(df.columns)} columns.\n")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 2. Schema Inspection
print("--- Detected Columns ---")
print(df.columns.tolist())

print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print("\n--- Sample Data (First 2 Rows) ---")
print(df.head(2).to_string())
