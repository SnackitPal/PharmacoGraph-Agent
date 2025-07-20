# scripts/build_smiles_map_utility.py (Final Version)

"""
This is a one-time utility script to build a definitive DrugBank ID -> SMILES
mapping. It uses the most robust method identified:
1. Downloads the massive PubChem CID-Synonym file.
2. Parses it to find all CIDs associated with DrugBank IDs in our dataset.
3. Uses efficient batch API calls to PubChem to get SMILES strings for those CIDs.
4. Saves the final mapping to a local JSON file.

This script should be run once to generate the 'drugbank_to_smiles.json' file.
After that, the main 'agent_molecular_features.py' will use the JSON file
for fast, reliable, offline enrichment of the knowledge graph.
"""

import gzip
import json
import requests
import time
import os
import pandas as pd
from pathlib import Path

class PubChemDrugBankMapper:
    """
    A class to manage the process of mapping DrugBank IDs to SMILES strings
    using a combination of PubChem's bulk files and batch API calls.
    """
    def __init__(self):
        self.drugbank_to_cid = {}
        self.cid_to_smiles = {}
        self.final_mapping = {}
        
    def download_synonym_file(self, output_path):
        """Downloads the PubChem CID-Synonym mapping file."""
        url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz"
        
        print(f"Downloading {url}...")
        print("This file is large (~2GB) and may take several minutes...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded to {output_path}")
            return output_path
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to download the file. Please check your network connection. Error: {e}")
            return None
    
    def parse_drugbank_synonyms(self, synonym_file_path):
        """Extracts DrugBank ID -> CID mappings from the synonym file."""
        print(f"Parsing DrugBank synonyms from {synonym_file_path}...")
        
        drugbank_count = 0
        
        with gzip.open(synonym_file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if (line_num + 1) % 5000000 == 0: # Progress update every 5 million lines
                    print(f"  ...processed {line_num + 1:,} lines, found {drugbank_count} DrugBank IDs...")
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cid, synonym = parts[0], parts[1]
                    
                    # Robust check for DrugBank ID format (e.g., DB00123)
                    if synonym.startswith('DB') and len(synonym) == 7 and synonym[2:].isdigit():
                        self.drugbank_to_cid[synonym] = cid
                        drugbank_count += 1
        
        print(f"Finished parsing. Found {drugbank_count} total DrugBank ID -> CID mappings.")
        return self.drugbank_to_cid
    
    # --- Corrected fetch_smiles_batch function ---
    def fetch_smiles_batch(self, cids):
        """Fetches SMILES for a batch of CIDs using an efficient batch API call."""
        cid_list = ','.join(cids)
        # The URL is correct - it requests the IsomericSMILES property
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}/property/IsomericSMILES/JSON"
        headers = {'User-Agent': 'PharmacoGraph-DrugBank-Mapper/1.0'}
        
        try:
            response = requests.get(url, timeout=60, headers=headers)
            if response.status_code == 200:
                data = response.json()
                for prop in data.get('PropertyTable', {}).get('Properties', []):
                    cid = str(prop['CID'])
                    # THIS IS THE CORRECTED LINE: The key is 'SMILES', not 'IsomericSMILES'
                    smiles = prop.get('SMILES')
                    if smiles:
                        self.cid_to_smiles[cid] = smiles
            else:
                print(f"  WARNING: Batch request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Batch request failed due to a network error: {e}")
        
        # Polite rate-limiting
        time.sleep(0.5)
    
    def build_final_mapping(self, your_drugbank_ids):
        """Builds the final DrugBank ID -> SMILES mapping for a specific list of IDs."""
        print(f"\nMatching the {len(your_drugbank_ids)} drugs from your dataset against the {len(self.drugbank_to_cid)} mappings found in the PubChem file...")
        
        cids_to_fetch = []
        missing_ids = []
        
        for db_id in your_drugbank_ids:
            if db_id in self.drugbank_to_cid:
                cids_to_fetch.append(self.drugbank_to_cid[db_id])
            else:
                missing_ids.append(db_id)
        
        print(f"Found CIDs for {len(cids_to_fetch)}/{len(your_drugbank_ids)} DrugBank IDs.")
        if missing_ids:
            print(f"Could not find a CID for {len(missing_ids)} IDs in the PubChem file. Sample missing IDs: {missing_ids[:10]}")
        
        print("\nFetching SMILES from PubChem in efficient batches...")
        batch_size = 100
        for i in range(0, len(cids_to_fetch), batch_size):
            batch = cids_to_fetch[i:i + batch_size]
            self.fetch_smiles_batch(batch)
            print(f"  Processed batch {i//batch_size + 1}/{(len(cids_to_fetch) + batch_size - 1)//batch_size}")
        
        # Create the final mapping
        for db_id in your_drugbank_ids:
            cid = self.drugbank_to_cid.get(db_id)
            if cid and cid in self.cid_to_smiles:
                self.final_mapping[db_id] = self.cid_to_smiles[cid]
        
        print(f"\nFinal mapping contains {len(self.final_mapping)} DrugBank ID -> SMILES pairs.")
        return self.final_mapping
    
    def save_mapping(self, output_path):
        """Saves the final mapping to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.final_mapping, f, indent=2)
        print(f"Saved final mapping to {output_path}")

def main():
    """Main execution function"""
    print("=== PubChem DrugBank Mapper Utility ===")
    
    # --- Define Paths ---
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    
    # --- Load your specific DrugBank IDs from your dataset ---
    print("Loading your dataset to get the list of required DrugBank IDs...")
    edge_list_path = DATA_DIR / 'drug_adr_edge_list.csv'
    try:
        df = pd.read_csv(edge_list_path)
        your_drugbank_ids = df['drugbank_id'].unique().tolist()
        print(f"Found {len(your_drugbank_ids)} unique DrugBank IDs in your dataset.")
    except FileNotFoundError:
        print(f"ERROR: Could not find your edge list at {edge_list_path}. Please ensure Agent 1 has been run successfully.")
        return
    
    # Initialize the mapper
    mapper = PubChemDrugBankMapper()
    
    # Download the giant synonym file to the data directory if it doesn't already exist
    synonym_file = DATA_DIR / "CID-Synonym-filtered.gz"
    if not os.path.exists(synonym_file):
        mapper.download_synonym_file(synonym_file)
    else:
        print(f"Using existing synonym file found at: {synonym_file}")
    
    # Parse the file to get DrugBank ID -> CID mappings
    mapper.parse_drugbank_synonyms(synonym_file)
    
    # Build the final mapping for your specific list of IDs
    mapper.build_final_mapping(your_drugbank_ids)
    
    # Save the final result to the data directory
    output_path = DATA_DIR / "drugbank_to_smiles.json"
    mapper.save_mapping(output_path)
    
    print("\n=== Summary ===")
    print(f"Successfully mapped {len(mapper.final_mapping)} DrugBank IDs to SMILES.")
    if your_drugbank_ids:
        coverage = 100 * len(mapper.final_mapping) / len(your_drugbank_ids)
        print(f"Coverage for your dataset: {len(mapper.final_mapping)}/{len(your_drugbank_ids)} = {coverage:.1f}%")
    
    print(f"\nYou can now use the generated file '{output_path}' in your main pipeline!")

if __name__ == "__main__":
    main()