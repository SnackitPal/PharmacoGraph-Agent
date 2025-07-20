# scripts/test_id_validity.py

import requests
import json
import pandas as pd
from pathlib import Path

# --- Configuration ---
# A list of DrugBank IDs to test.
# We include our known-good ID and the first few from our actual dataset.
known_good_id = 'DB00316' # Acetaminophen

# Load the first few drug IDs from our actual dataset
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
EDGE_LIST_PATH = DATA_DIR / 'drug_adr_edge_list.csv'
edge_list = pd.read_csv(EDGE_LIST_PATH)
dataset_ids = edge_list['drugbank_id'].unique()[:5].tolist() # Get the first 5 unique IDs

# Combine them for the test
drug_ids_to_test = [known_good_id] + dataset_ids

headers = {'User-Agent': 'PharmacoGraph-Agent-API-Inspector/1.0'}

print("--- Starting API Response Inspection ---")
print(f"Testing the following IDs: {drug_ids_to_test}\n")

for drug_id in drug_ids_to_test:
    print(f"--- Testing DrugBank ID: {drug_id} ---")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drug_id}/cids/JSON"
    print(f"Querying URL: {url}")
    
    try:
        response = requests.get(url, timeout=15, headers=headers)
        print(f"Response Status Code: {response.status_code}")
        
        # We will print the raw text to see exactly what the server sent back
        print(f"Raw Response Text: {response.text.strip()}")
        
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")

print("\n--- Inspection Complete ---")