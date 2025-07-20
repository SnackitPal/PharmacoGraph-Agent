# scripts/agent_3b_feature_calculator.py
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
import json
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
GRAPH_PATH = DATA_DIR / 'knowledge_graph.graphml'
SMILES_MAP_PATH = DATA_DIR / 'drugbank_to_smiles.json'

def main():
    print("--- Agent 3B: Feature Calculator START ---")

    # Load the graph and the SMILES map
    print(f"Loading graph from: {GRAPH_PATH}")
    G = nx.read_graphml(GRAPH_PATH)
    
    print(f"Loading SMILES map from: {SMILES_MAP_PATH}")
    with open(SMILES_MAP_PATH, 'r') as f:
        smiles_map = json.load(f)

    # Enrich the drug nodes
    print("Enriching drug nodes with molecular features...")
    drug_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'drug']
    total_drugs = len(drug_nodes)
    enriched_count = 0

    for drug_id in drug_nodes:
        smiles = smiles_map.get(drug_id)
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    G.nodes[drug_id]['mol_weight'] = Descriptors.MolWt(mol)
                    G.nodes[drug_id]['logp'] = Descriptors.MolLogP(mol)
                    # ... add any other descriptors you want here ...
                    enriched_count += 1
            except Exception as e:
                print(f"  [RDKit Error] for {drug_id}: {e}")

    print(f"Successfully enriched {enriched_count}/{total_drugs} drug nodes.")

    # Save the final, enriched graph
    print(f"Saving enriched graph back to: {GRAPH_PATH}")
    nx.write_graphml(G, GRAPH_PATH)

    print("--- Agent 3B: Feature Calculator COMPLETE ---")

if __name__ == '__main__':
    main()