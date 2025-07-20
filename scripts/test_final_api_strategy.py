# scripts/test_final_api_strategy.py

import pandas as pd
import requests
import gc # Garbage Collection library
from pathlib import Path

print("--- Starting Final API Strategy Diagnostic Test ---")

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
EDGE_LIST_PATH = DATA_DIR / 'drug_adr_edge_list.csv'

try:
    # --- Step 1: Simulate the full agent's memory load ---
    print(f"Loading large DataFrame from: {EDGE_LIST_PATH}")
    edge_list_df = pd.read_csv(EDGE_LIST_PATH)
    # Get a small sample of real DrugBank IDs from our dataset
    drugbank_ids = edge_list_df['drugbank_id'].unique()[:10]
    print(f"Successfully loaded DataFrame. Testing with first {len(drugbank_ids)} IDs.")

    # --- Step 2: Force Garbage Collection (as suggested by LLM) ---
    print("Forcing garbage collection...")
    gc.collect()

    # --- Step 3: Create a fresh requests session (as suggested by LLM) ---
    print("Creating a new requests.Session object...")
    session = requests.Session()
    headers = {'User-Agent': 'PharmacoGraph-Agent-Final-Test/1.0'}
    session.headers.update(headers)

    # --- Step 4: Test the NEW API endpoint for each ID ---
    print("\n--- Testing the new API endpoint... ---")
    for test_id in drugbank_ids:
        # This is the new, more direct URL endpoint
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/DrugBank/{test_id}/JSON"
        print(f"Testing ID: {test_id} at URL: {url}")
        
        try:
            response = session.get(url, timeout=15)
            # We will print the status and the raw text to see exactly what we get back
            print(f"  -> Status: {response.status_code}, Response Text: {response.text.strip()}")
        except requests.exceptions.RequestException as e:
            print(f"  -> FAILED with network error: {e}")

except FileNotFoundError:
    print(f"ERROR: Could not find the edge list file at {EDGE_LIST_PATH}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- Diagnostic Complete ---")