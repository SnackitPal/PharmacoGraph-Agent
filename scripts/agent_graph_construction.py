# scripts/agent_graph_construction.py

"""
Agent 2: Knowledge Graph Constructor

This agent takes the clean edge list produced by Agent 1 and constructs
a graph using the NetworkX library. It adds metadata to the nodes and
saves the final graph object for further analysis or use by other agents.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import time

# --- 1. CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_PATH = DATA_DIR / 'drug_adr_edge_list.csv'
OUTPUT_PATH = DATA_DIR / 'knowledge_graph.graphml' # GraphML is a standard format for saving graphs

# --- 2. MAIN WORKFLOW ---
def main():
    """
    Main function to orchestrate the graph construction.
    """
    start_time = time.time()
    print("--- Knowledge Graph Constructor Agent: START ---")

    # --- Load the Edge List ---
    print(f"Loading edge list from: {INPUT_PATH}")
    edge_list_df = pd.read_csv(INPUT_PATH)

    # --- Create the Graph ---
    # We can create a graph directly from the pandas DataFrame edge list!
    print("Constructing graph from edge list...")
    G = nx.from_pandas_edgelist(
        edge_list_df,
        source='drugbank_id',
        target='reaction',
        create_using=nx.Graph() # Use a standard, undirected graph for now
    )
    print("Graph constructed successfully.")

    # --- Add Node Attributes ---
    # This is a crucial step for GNNs. We need to tell the graph
    # which nodes are drugs and which are reactions.
    print("Adding node attributes (e.g., 'type')...")
    for node in G.nodes():
        if node.startswith('DB'):
            G.nodes[node]['type'] = 'drug'
        else:
            G.nodes[node]['type'] = 'reaction'
            
    # --- Perform Basic Graph Analysis ---
    print("\n--- Basic Graph Analysis ---")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # Calculate the degree of each node (how many connections it has)
    degrees = dict(G.degree())
    # Find the top 5 most connected nodes
    top_5_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:5]
    
    print("\nTop 5 most connected nodes (Highest Degree):")
    for node, degree in top_5_nodes:
        node_type = G.nodes[node]['type']
        print(f"  - Node: {node} (Type: {node_type}), Degree: {degree}")

    # --- Save the Graph ---
    print(f"\nSaving graph to: {OUTPUT_PATH}")
    nx.write_graphml(G, OUTPUT_PATH)

    print("\n--- Knowledge Graph Constructor Agent: COMPLETE ---")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()