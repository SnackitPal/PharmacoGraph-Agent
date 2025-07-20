# scripts/test_api_mapping_completeness.py

import requests
import json
import pandas as pd
from pathlib import Path

# --- Configuration ---
# A list of DrugBank IDs to test, mixing known-good, dataset-specific, and dummy IDs.
# As suggested by the LLM.
ids_to_test = [
    "DB00316",  # Known good (Acetaminophen)
    "DB01006",  # First ID from our dataset
    "DB01259",  # Second ID from our dataset
    "DB00947",  # Third ID from our dataset
    "DB12345",  # A likely non-existent ID
    "DB99999"   # A dummy ID
]

headers = {'User-Agent': 'PharmacoGraph-Agent-Mapping-Test/1.0'}

print("--- Starting API Mapping Completeness Test ---")
print(f"Testing the following IDs: {ids_to_test}\n")

for drug_id in ids_to_test:
    print(f"--- Testing DrugBank ID: {drug_id} ---")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drug_id}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=15, headers=headers)
        print(f"Response Status Code: {response.status_code}")
        
        # We will print the raw text to see exactly what the server sent back
        print(f"Raw Response Text: {response.text.strip()}")
        
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")

print("\n--- Test Complete ---")