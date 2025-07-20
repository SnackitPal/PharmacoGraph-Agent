# scripts/test_all_endpoints.py

import requests
import urllib.parse

# --- Configuration ---
# We'll use a drug we know is in our dataset and is common.
# We'll use the standardized name, not the messy one.
# Let's use 'Methotrexate' which maps to DB00563
drug_name = "Methotrexate"
drugbank_id = "DB00563" 

headers = {'User-Agent': 'PharmacoGraph-Agent-Endpoint-Test/1.0'}

print(f"--- Starting Final Endpoint Gauntlet Test for '{drug_name}' ({drugbank_id}) ---")

# --- Test A: Direct Name Search ---
print("\n--- Testing Option A: Direct Name Search ---")
# We must URL-encode the name to handle spaces or special characters
encoded_name = urllib.parse.quote(drug_name)
url_a = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES/JSON"
print(f"Querying URL: {url_a}")
try:
    response_a = requests.get(url_a, timeout=15, headers=headers)
    print(f"  -> Status: {response_a.status_code}, Response: {response_a.text.strip()}")
except Exception as e:
    print(f"  -> FAILED with error: {e}")

# --- Test B: Synonym Search ---
# Note: The LLM suggests using the DrugBank ID here, but this endpoint is typically for synonyms.
# We will test it exactly as written.
print("\n--- Testing Option B: Synonym Search (using DrugBank ID) ---")
url_b = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/synonym/{drugbank_id}/property/IsomericSMILES/JSON"
print(f"Querying URL: {url_b}")
try:
    response_b = requests.get(url_b, timeout=15, headers=headers)
    print(f"  -> Status: {response_b.status_code}, Response: {response_b.text.strip()}")
except Exception as e:
    print(f"  -> FAILED with error: {e}")

# --- Test C: The CID-based approach (This is our original successful method) ---
print("\n--- Testing Option C: The Proven CID-based approach ---")
# This test will confirm our previous findings.
cid = None
try:
    # First, get the CID
    print("Step 1: Getting CID from DrugBank ID...")
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drugbank_id}/cids/JSON"
    cid_response = requests.get(cid_url, timeout=15, headers=headers)
    cid = cid_response.json().get('InformationList', {}).get('Information', [{}])[0].get('CID', [None])[0]
    print(f"  -> Found CID: {cid}")

    # Second, use the CID to get the SMILES
    if cid:
        print("Step 2: Getting SMILES from CID...")
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
        smiles_response = requests.get(smiles_url, timeout=15, headers=headers)
        print(f"  -> Status: {smiles_response.status_code}, Response: {smiles_response.text.strip()}")
    else:
        print("  -> Could not find CID, skipping SMILES lookup.")
except Exception as e:
    print(f"  -> FAILED with error: {e}")

print("\n--- Test Complete ---")