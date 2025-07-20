# scripts/agent_3_smiles_fetcher.py (Bulletproof Version)

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

def get_smiles_from_drugbank_id_direct(drugbank_id):
    """
    This function is based on the proven logic from our successful diagnostic tests.
    It performs a direct API call to PubChem to resolve a DrugBank ID.
    """
    headers = {'User-Agent': 'PharmacoGraph-Agent/1.0'}
    
    try:
        # Step 1: Use the proven endpoint to get the PubChem CID
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drugbank_id}/cids/JSON"
        response = requests.get(cid_url, timeout=15, headers=headers)
        
        # Check for a 404 error for non-existent IDs
        if response.status_code == 404:
            return None
            
        response.raise_for_status()
        data = response.json()
        
        # Use the JSON structure we observed in the successful test
        cids = data.get('InformationList', {}).get('Information', [{}])[0].get('CID', [])
        if not cids:
            return None
        
        # Step 2: Use the CID to get the SMILES string
        cid = cids[0]
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
        smiles_response = requests.get(smiles_url, timeout=15, headers=headers)
        smiles_response.raise_for_status()
        
        return smiles_response.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('IsomericSMILES')
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError, KeyError):
        # Catch any possible error during the process and return None
        return None

def main():
    print("--- Agent 3: SMILES Fetcher (Bulletproof Version) START ---")
    
    # Load only the column we need to get the unique list of drugs
    edge_list = pd.read_csv(EDGE_LIST_PATH, usecols=['drugbank_id'])
    unique_drug_ids = edge_list['drugbank_id'].unique()
    total_drugs = len(unique_drug_ids)
    
    smiles_map = {}
    
    print(f"Fetching SMILES for {total_drugs} unique drugs...")
    for i, drug_id in enumerate(unique_drug_ids):
        if (i + 1) % 50 == 0:
            print(f"  Processing drug {i+1}/{total_drugs}...")
        
        smiles = get_smiles_from_drugbank_id_direct(drug_id)
        if smiles:
            smiles_map[drug_id] = smiles
            
        # Polite rate-limiting is crucial
        time.sleep(0.2)
        
    print(f"\nSuccessfully fetched SMILES for {len(smiles_map)}/{total_drugs} drugs.")
    
    if len(smiles_map) > 0:
        print(f"Saving SMILES map to: {SMILES_MAP_PATH}")
        with open(SMILES_MAP_PATH, 'w') as f:
            json.dump(smiles_map, f, indent=2)
            
        print("\n--- Verification: Sample from SMILES map ---")
        # Print a few examples from our new dictionary
        for i, (drug, smiles) in enumerate(smiles_map.items()):
            if i >= 5: break
            print(f"  {drug}: {smiles}")
    else:
        print("\n--- WARNING: Final map is empty. The Heisenbug persists. ---")
        
    print("\n--- Agent 3: SMILES Fetcher COMPLETE ---")

if __name__ == '__main__':
    main()