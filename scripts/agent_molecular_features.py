# scripts/agent_molecular_features.py (Final Offline Version)

"""
Agent 3: Molecular Feature Agent

This agent enriches the knowledge graph by adding molecular descriptors to drug nodes.
It uses the pre-compiled JSON map (created by the utility script) to find the
chemical structure (SMILES) for each drug, calculates features using RDKit,
and saves the enriched graph.
"""
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
import json
from pathlib import Path
import time

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
GRAPH_PATH = DATA_DIR / 'knowledge_graph.graphml'
SMILES_MAP_PATH = DATA_DIR / 'drugbank_to_smiles.json'

def main():
    start_time = time.time()
    print("--- Molecular Feature Agent: START ---")

    print(f"Loading graph from: {GRAPH_PATH}")
    G = nx.read_graphml(GRAPH_PATH)
    
    print(f"Loading SMILES map from: {SMILES_MAP_PATH}")
    with open(SMILES_MAP_PATH, 'r') as f:
        smiles_map = json.load(f)

    print("Enriching drug nodes with molecular features...")
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'drug']
    enriched_count = 0
    for drug_id in drug_nodes:
        smiles = smiles_map.get(drug_id)
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Calculate and add descriptors as node attributes
                    G.nodes[drug_id]['mol_weight'] = Descriptors.MolWt(mol)
                    G.nodes[drug_id]['logp'] = Descriptors.MolLogP(mol)
                    G.nodes[drug_id]['h_bond_donors'] = Descriptors.NumHDonors(mol)
                    G.nodes[drug_id]['h_bond_acceptors'] = Descriptors.NumHAcceptors(mol)
                    G.nodes[drug_id]['tpsa'] = Descriptors.TPSA(mol)
                    enriched_count += 1
            except Exception as e:
                print(f"  [RDKit Error] for {drug_id}: {e}")

    print(f"Successfully enriched {enriched_count}/{len(drug_nodes)} drug nodes.")

    print(f"Saving enriched graph back to: {GRAPH_PATH}")
    nx.write_graphml(G, GRAPH_PATH)
    
    print("\n--- Verification: Attributes of drug DB00316 (Acetaminophen) ---")
    if 'DB00316' in G.nodes:
        print(G.nodes['DB00316'])

    print(f"\n--- Molecular Feature Agent: COMPLETE in {time.time() - start_time:.2f}s ---")

if __name__ == '__main__':
    main()