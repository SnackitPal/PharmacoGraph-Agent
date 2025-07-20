# scripts/agent_3a_smiles_fetcher.py (Diagnostic Version)
import pandas as pd
import requests
import time
import json
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
EDGE_LIST_PATH = DATA_DIR / 'drug_adr_edge_list.csv'
SMILES_MAP_PATH = DATA_DIR / 'drugbank_to_smiles.json'

# --- API call function with ERROR PRINTING ---
def get_smiles_from_drugbank_id_direct(drugbank_id, retries=3, delay=1):
    headers = {'User-Agent': 'PharmacoGraph-Agent/1.0'}
    for attempt in range(retries):
        try:
            cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drugbank_id}/cids/JSON"
            response = requests.get(cid_url, timeout=15, headers=headers)
            response.raise_for_status()
            cids = response.json().get('IdentifierList', {}).get('CID', [])
            if not cids: return None
            
            cid = cids[0]
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
            smiles_response = requests.get(smiles_url, timeout=15, headers=headers)
            smiles_response.raise_for_status()
            return smiles_response.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('IsomericSMILES')
            
        except requests.exceptions.RequestException as e:
            # THIS IS THE CRITICAL CHANGE: WE WILL NOW PRINT THE ERROR
            print(f"  [API Request Error] for {drugbank_id}: {e}")
            time.sleep(delay)
    return None

def main():
    print("--- Agent 3A: SMILES Fetcher (DIAGNOSTIC MODE) START ---")
    
    edge_list = pd.read_csv(EDGE_LIST_PATH)
    unique_drug_ids = edge_list['drugbank_id'].unique()
    # We will only test the first 25 drugs to see the error pattern quickly.
    unique_drug_ids = unique_drug_ids[:25]
    total_drugs = len(unique_drug_ids)
    
    smiles_map = {}
    
    print(f"Fetching SMILES for the first {total_drugs} unique drugs...")
    for i, drug_id in enumerate(unique_drug_ids):
        smiles = get_smiles_from_drugbank_id_direct(drug_id)
        if smiles:
            smiles_map[drug_id] = smiles
        time.sleep(0.2)
        
    print(f"\nSuccessfully fetched SMILES for {len(smiles_map)}/{total_drugs} drugs.")
    print("--- Agent 3A: SMILES Fetcher (DIAGNOSTIC MODE) COMPLETE ---")

if __name__ == '__main__':
    main()