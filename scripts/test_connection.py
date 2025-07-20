# scripts/test_connection.py

import requests
import time

print("--- Starting Minimal Connection Test ---")

# The exact DrugBank ID and URL that we know works with curl
drugbank_id = 'DB00316'
cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RegistryID/{drugbank_id}/cids/JSON"

# The standard User-Agent header
headers = {
    'User-Agent': 'PharmacoGraph-Agent-Connection-Test/1.0'
}

print(f"Attempting to connect to: {cid_url}")

try:
    # Make the request with a 15-second timeout
    response = requests.get(cid_url, timeout=15, headers=headers)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # If we get here, the connection worked!
    print("\n--- ✅ SUCCESS! ---")
    print(f"Status Code: {response.status_code}")
    print("Successfully received data from the PubChem server.")
    print(f"Response JSON: {response.json()}")

except requests.exceptions.RequestException as e:
    # If we get here, the connection failed.
    print("\n--- ❌ FAILURE! ---")
    print("The connection failed. This confirms a network-level issue with Python/Requests.")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")