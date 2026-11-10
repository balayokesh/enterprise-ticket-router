import pandas as pd
import numpy as np

def inspect_data(input_path="data.csv"):
    """Loads and inspects the raw dataset schema and missing values."""
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded '{input_path}' with {len(df)} rows and {len(df.columns)} columns.\n")
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

    print("--- Detected Columns ---")
    print(df.columns.tolist())

    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())

    print("\n--- Sample Data (First 2 Rows) ---")
    print(df.head(2).to_string())
    
    return df

def clean_support_data(input_path="data.csv", output_path="clean_data.csv"):
    """Cleans raw support ticket data and saves the output."""
    print("\nStarting data cleaning pipeline...")
    df = pd.read_csv(input_path)
    
    # 1. Fill missing text fields for unresolved tickets
    df['Resolution'] = df['Resolution'].fillna("Pending Resolution")
    df['First Response Time'] = df['First Response Time'].fillna("Not Recorded")
    df['Time to Resolution'] = df['Time to Resolution'].fillna("N/A")
    df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].fillna(0)
    
    # 2. Clean text escape characters and extra spaces in descriptions
    df['Ticket Description'] = df['Ticket Description'].astype(str).str.replace(r'\r\n', ' ', regex=True)
    df['Ticket Description'] = df['Ticket Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 3. Strip whitespace from categorical columns
    categorical_cols = ['Ticket Type', 'Ticket Subject', 'Ticket Priority', 'Ticket Channel']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Data cleaning complete! Saved processed data to {output_path}. Total rows: {len(df)}")

if __name__ == "__main__":
    inspect_data()
    clean_support_data()
