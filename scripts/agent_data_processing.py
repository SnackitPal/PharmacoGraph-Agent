import pandas as pd
from pathlib import Path

# Define the root directory of the project
# This allows for consistent pathing whether running from the root or the scripts directory
try:
    # Assumes the script is run from the root directory, which is standard
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Fallback for interactive environments like Jupyter or when __file__ is not defined
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / 'data'

def load_data():
    """
    Loads all raw data files from the data directory and returns them as pandas DataFrames.

    This function encapsulates the data loading logic from the exploratory notebook,
    providing a clean interface for the main processing script.

    Returns:
        tuple: A tuple containing the following DataFrames in order:
               - sider_df (pd.DataFrame): SIDER side effects data.
               - drug_df (pd.DataFrame): FAERS drug data.
               - reac_df (pd.DataFrame): FAERS reaction data.
               - indications_df (pd.DataFrame): SIDER drug indications data.
    """
    # --- 1. Define File Paths ---
    # Construct full paths to the data files using the DATA_DIR constant
    sider_effects_path = DATA_DIR / 'sider_side_effects.tsv'
    sider_indications_path = DATA_DIR / 'sider_indications.tsv'
    faers_demo_path = DATA_DIR / 'ASCII' / 'DEMO25Q1.txt'
    faers_drug_path = DATA_DIR / 'ASCII' / 'DRUG25Q1.txt'
    faers_reac_path = DATA_DIR / 'ASCII' / 'REAC25Q1.txt'

    # --- 2. Load DataFrames ---
    print("--- Loading SIDER data... ---")
    sider_df = pd.read_csv(sider_effects_path, sep='\t')

    # Load indications data with no header and assign column names
    indications_df = pd.read_csv(sider_indications_path, sep='\t', header=None)
    indications_df = indications_df.rename(columns={0: 'drugbank_id', 1: 'indication_name'})

    print("--- Loading FAERS data (DEMO, DRUG, REAC)... ---")
    # Use low_memory=False to prevent DtypeWarning from mixed types
    demo_df = pd.read_csv(faers_demo_path, sep='$', low_memory=False)
    drug_df = pd.read_csv(faers_drug_path, sep='$', low_memory=False)
    reac_df = pd.read_csv(faers_reac_path, sep='$', low_memory=False)

    print(f"--- SIDER data loaded successfully. Shape: {sider_df.shape} ---")

    # --- 3. Return DataFrames ---
    # The notebook merges drug_df and reac_df, but for now, we return them raw
    return sider_df, drug_df, reac_df, indications_df

def main():
    """
    Main function to orchestrate the data processing pipeline.
    """
    print("--- Starting Data Processing Pipeline ---")

    # --- Load Data ---
    sider_df, drug_df, reac_df, indications_df = load_data()

    # --- (Further processing steps will be added here) ---

    print("\n--- Data Loading Complete ---")
    print(f"SIDER DataFrame shape: {sider_df.shape}")
    print(f"FAERS Drug DataFrame shape: {drug_df.shape}")
    print(f"FAERS Reaction DataFrame shape: {reac_df.shape}")
    print(f"Indications DataFrame shape: {indications_df.shape}")

    print("\n--- Data Processing Pipeline Finished ---")

if __name__ == '__main__':
    main()
