# scripts/agent_reporting.py (FINAL VERSION with Step-by-Step Status)

import torch
import networkx as nx
import streamlit as st
from pathlib import Path
import pandas as pd
import time

# --- (The GNN Model, Decoder, and Model classes are the same as before) ---
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

class Decoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['drug'][row], z_dict['reaction'][col]], dim=-1)
        z = self.lin1(z).relu(); z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data_metadata, drug_features_dim, reaction_features_dim):
        super().__init__()
        self.drug_lin = torch.nn.Linear(drug_features_dim, hidden_channels)
        self.reaction_lin = torch.nn.Linear(reaction_features_dim, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data_metadata, aggr='sum')
        self.decoder = Decoder(hidden_channels)
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {'drug': self.drug_lin(x_dict['drug']),'reaction': self.reaction_lin(x_dict['reaction'])}
        z_dict = self.gnn(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# --- Main Agent Functions ---

@st.cache_resource
def load_artifacts():
    """Loads all necessary data and the trained model."""
    DATA_DIR = Path('../data')
    GRAPH_PATH = DATA_DIR / 'knowledge_graph.graphml'
    MODEL_PATH = DATA_DIR / 'best_gnn_model.pt'
    SIDER_EFFECTS_PATH = DATA_DIR / 'sider_side_effects.tsv'

    G = nx.read_graphml(GRAPH_PATH)
    data = HeteroData()
    
    sider_df = pd.read_csv(SIDER_EFFECTS_PATH, sep='\t')
    name_map_df = sider_df[['drugbank_id', 'drugbank_name']].drop_duplicates()
    drug_id_to_name = dict(zip(name_map_df['drugbank_id'], name_map_df['drugbank_name']))
    drug_name_to_id = dict(zip(name_map_df['drugbank_name'], name_map_df['drugbank_id']))

    drug_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']
    drug_map = {node_id: i for i, node_id in enumerate(drug_nodes)}
    rev_drug_map = {i: node_id for node_id, i in drug_map.items()}
    
    reaction_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'reaction']
    reaction_map = {node_id: i for i, node_id in enumerate(reaction_nodes)}
    rev_reaction_map = {i: node_id for node_id, i in reaction_map.items()}

    feature_keys = ['mol_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors', 'tpsa']
    drug_features = [[float(d.get(f, 0)) for f in feature_keys] for n, d in G.nodes(data=True) if d['type'] == 'drug']
    data['drug'].x = torch.tensor(drug_features, dtype=torch.float)
    data['reaction'].x = torch.eye(len(reaction_nodes))
    
    src = [drug_map[u] for u, v in G.edges() if G.nodes[u]['type']=='drug' and u in drug_map and v in reaction_map]
    dst = [reaction_map[v] for u, v in G.edges() if G.nodes[u]['type']=='drug' and u in drug_map and v in reaction_map]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data['drug', 'causes', 'reaction'].edge_index = edge_index
    data['reaction', 'rev_causes', 'drug'].edge_index = edge_index.flip([0])
    
    model = Model(
        hidden_channels=64, 
        data_metadata=data.metadata(),
        drug_features_dim=data['drug'].x.shape[1],
        reaction_features_dim=data['reaction'].x.shape[1]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    return model, data, drug_map, rev_drug_map, rev_reaction_map, drug_id_to_name, drug_name_to_id

@torch.no_grad()
def predict_adrs(model, data, patient_drug_ids, drug_map, rev_drug_map, rev_reaction_map, status_indicator):
    """Makes predictions for a given list of drugs against all possible reactions."""
    status_indicator.update(label="Step 1/3: Generating node embeddings with GNN...")
    if not hasattr(data, 'edge_index_dict'):
        data.edge_index_dict = { k: v.edge_index for k, v in data.edge_items() }
    x_dict = {'drug': model.drug_lin(data['drug'].x), 'reaction': model.reaction_lin(data['reaction'].x)}
    z_dict = model.gnn(x_dict, data.edge_index_dict)
    
    patient_drug_indices = [drug_map[drug_id] for drug_id in patient_drug_ids if drug_id in drug_map]
    num_reactions = data['reaction'].x.shape[0]
    all_predictions = []
    
    status_indicator.update(label="Step 2/3: Predicting links for patient's drugs...")
    time.sleep(0.5) # Small delay for UI effect
    
    for drug_idx in patient_drug_indices:
        drug_tensor = torch.tensor([drug_idx] * num_reactions)
        reaction_tensor = torch.arange(num_reactions)
        edge_label_index = torch.stack([drug_tensor, reaction_tensor], dim=0)
        pred = model.decoder(z_dict, edge_label_index).sigmoid()
        
        for i, p in enumerate(pred):
            reaction_name = rev_reaction_map[i]
            drug_id = rev_drug_map[drug_idx]
            all_predictions.append((drug_id, reaction_name, p.item()))
            
    status_indicator.update(label="Step 3/3: Ranking predictions...")
    time.sleep(0.5) # Small delay for UI effect
            
    return pd.DataFrame(all_predictions, columns=['Drug_ID', 'Predicted ADR', 'Probability'])

# --- Streamlit User Interface (with Step-by-Step Status) ---
def main():
    st.set_page_config(page_title="PharmacoGraph ADR Predictor", layout="wide")
    st.title("🧪 PharmacoGraph-Agent: ADR Predictor")
    
    model, data, drug_map, rev_drug_map, rev_reaction_map, drug_id_to_name, drug_name_to_id = load_artifacts()

    st.header("Patient Profile")
    
    available_drug_names = sorted([drug_id_to_name[db_id] for db_id in drug_map.keys() if db_id in drug_id_to_name])
    
    selected_drug_names = st.multiselect(
        'Search and select drugs for the patient regimen:',
        options=available_drug_names,
        default=['Methotrexate', 'Acetaminophen']
    )

    if st.button('Predict ADRs'):
        if not selected_drug_names:
            st.warning("Please select at least one drug.")
        else:
            patient_drug_ids = [drug_name_to_id[name] for name in selected_drug_names]
            
            # NEW: Use st.status to show the backend process
            with st.status("Running GNN model to predict ADRs...", expanded=True) as status:
                predictions_df = predict_adrs(model, data, patient_drug_ids, drug_map, rev_drug_map, rev_reaction_map, status)
                status.update(label="Prediction complete!", state="complete", expanded=False)
            
            st.header("Top Predicted Adverse Drug Reactions")
            st.info("This table shows the top 20 predicted ADRs for the selected drugs, ranked by probability.")
            
            top_predictions = predictions_df.sort_values(by='Probability', ascending=False).head(20)
            
            top_predictions['Drug Name'] = top_predictions['Drug_ID'].map(drug_id_to_name)
            display_df = top_predictions[['Drug Name', 'Predicted ADR', 'Probability', 'Drug_ID']]
            
            st.dataframe(display_df.style.format({'Probability': '{:.2%}'}))

if __name__ == '__main__':
    main()