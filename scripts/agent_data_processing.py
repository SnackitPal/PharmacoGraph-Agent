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
OUTPUT_PATH = DATA_DIR / 'faers_drug_reaction_edges.csv'

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

def merge_faers_data(drug_df, reac_df):
    """
    Merges the FAERS drug and reaction DataFrames into a single, clean DataFrame.

    Args:
        drug_df (pd.DataFrame): The FAERS drug data.
        reac_df (pd.DataFrame): The FAERS reaction data.

    Returns:
        pd.DataFrame: The merged and cleaned FAERS DataFrame.
    """
    print("--- Merging FAERS drug and reaction tables... ---")

    # Perform an inner merge on the primaryid column
    faers_df = pd.merge(drug_df, reac_df, on='primaryid', how='inner')

    # Select and rename the key columns
    faers_df = faers_df[['primaryid', 'drugname', 'pt']]
    faers_df = faers_df.rename(columns={'pt': 'reaction'})

    print(f"--- FAERS data merged successfully. Shape: {faers_df.shape} ---")

    return faers_df

def clean_reactions(faers_df, sider_df):
    """
    Cleans the reaction data in the FAERS DataFrame by filtering against SIDER
    and a manual blocklist.

    Args:
        faers_df (pd.DataFrame): The merged FAERS data with a 'reaction' column.
        sider_df (pd.DataFrame): The SIDER data with a 'side_effect_name' column.

    Returns:
        pd.DataFrame: The cleaned FAERS DataFrame.
    """
    print("--- Cleaning reactions against SIDER ground truth... ---")
    print(f"Shape of FAERS data before SIDER filtering: {faers_df.shape}")

    # Create a set of known side effects from SIDER for efficient lookup
    known_side_effects = set(sider_df['side_effect_name'].str.lower())

    # Filter FAERS data
    # Keep rows where the reaction (in lowercase) is in the set of known side effects
    faers_cleaned_df = faers_df[faers_df['reaction'].str.lower().isin(known_side_effects)].copy()

    print(f"Shape of FAERS data after SIDER filtering: {faers_cleaned_df.shape}")

    # --- Stage 2: Manual Blocklist Filtering ---
    print("\n--- Applying manual blocklist to cleaned reactions... ---")
    print(f"Shape of FAERS data before blocklist filtering: {faers_cleaned_df.shape}")

    # Define a blocklist of common noise terms found during EDA
    reaction_blocklist = [
        'drug ineffective', 'condition aggravated', 'off label use',
        'product use in unapproved indication', 'death', 'wrong drug administered'
    ]

    # Filter out reactions that are in the blocklist
    faers_cleaned_df = faers_cleaned_df[~faers_cleaned_df['reaction'].str.lower().isin(reaction_blocklist)].copy()

    print(f"Shape of FAERS data after blocklist filtering: {faers_cleaned_df.shape}")


    return faers_cleaned_df

def standardize_drugs(faers_cleaned_df, sider_df, indications_df):
    """
    Standardizes drug names in the FAERS DataFrame to DrugBank IDs.

    Args:
        faers_cleaned_df (pd.DataFrame): The FAERS DataFrame after reaction cleaning.
        sider_df (pd.DataFrame): The SIDER data.
        indications_df (pd.DataFrame): The SIDER indications data.

    Returns:
        pd.DataFrame: The FAERS DataFrame with standardized DrugBank IDs.
    """
    print("\n--- Standardizing drug names to DrugBank IDs... ---")

    # --- 1. Build Synonym Dictionary ---
    # Extract drug names and IDs from both SIDER and indications data
    sider_names = sider_df[['drugbank_id', 'drugbank_name']].rename(columns={'drugbank_name': 'name'})
    # Corrected column name from 'indication_name' to 'drug_name' based on previous error
    indications_names = indications_df[['drugbank_id', 'indication_name']].rename(columns={'indication_name': 'name'})

    # Combine, remove duplicates, and drop any rows with missing data
    master_mapping_df = pd.concat([sider_names, indications_names]).drop_duplicates().dropna()

    # Create the synonym dictionary: uppercase drug name -> drugbank_id
    drug_synonyms = dict(zip(master_mapping_df['name'].str.upper(), master_mapping_df['drugbank_id']))
    print(f"Created a synonym dictionary with {len(drug_synonyms)} entries.")

    # --- 2. Apply Dictionary and Filter ---
    # Map drug names to DrugBank IDs
    faers_cleaned_df['drugbank_id'] = faers_cleaned_df['drugname'].str.upper().map(drug_synonyms)

    # Log mapping success rate
    mapped_count = faers_cleaned_df['drugbank_id'].notna().sum()
    total_count = len(faers_cleaned_df)
    mapping_rate = (mapped_count / total_count) * 100
    print(f"Successfully mapped {mapped_count} of {total_count} drug names ({mapping_rate:.2f}%).")

    # Filter out unmapped drugs
    faers_standardized_df = faers_cleaned_df.dropna(subset=['drugbank_id']).copy()
    print(f"Shape of DataFrame after dropping unmapped drugs: {faers_standardized_df.shape}")

    return faers_standardized_df

def create_and_save_edge_list(final_df, output_path):
    """
    Creates and saves the final edge list from the processed DataFrame.

    Args:
        final_df (pd.DataFrame): The fully processed and standardized DataFrame.
        output_path (str or Path): The path to save the final CSV file.
    """
    print("\n--- Creating and saving the final edge list... ---")

    # Select the final columns and drop duplicates
    edge_list_df = final_df[['drugbank_id', 'reaction']].drop_duplicates()

    # Save the edge list to CSV
    edge_list_df.to_csv(output_path, index=False)

    print(f"--- Found {len(edge_list_df)} unique drug-reaction edges. ---")
    print(f"--- Final edge list saved to: {output_path} ---")


def main():
    """
    Main function to orchestrate the data processing pipeline.
    """
    print("--- Starting Data Processing Pipeline ---")

    # --- Load Data ---
    sider_df, drug_df, reac_df, indications_df = load_data()

    # --- Merge FAERS Data ---
    faers_df = merge_faers_data(drug_df, reac_df)

    # --- Clean Reactions ---
    faers_cleaned_df = clean_reactions(faers_df, sider_df)

    # --- Standardize Drug Names ---
    faers_standardized_df = standardize_drugs(faers_cleaned_df, sider_df, indications_df)

    # --- Create and Save Final Edge List ---
    create_and_save_edge_list(faers_standardized_df, OUTPUT_PATH)

    print("\n--- Data Processing Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main()
