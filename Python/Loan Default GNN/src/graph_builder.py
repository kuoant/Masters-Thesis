import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from .constants import *

class GraphBuilder:
    @staticmethod
    def build_graph(X_train_subset, y_train_subset, connection_percentage=0.05):
        combined = X_train_subset.copy()
        combined['Default'] = y_train_subset.values
        
        # Filter default rows with dependents
        default_rows = combined[combined['Default'] == 1]
        has_dependents = default_rows[default_rows['HasDependents'] == 1]
        
        # Create adjacency matrix
        adj_matrix = pd.DataFrame(0, index=has_dependents.index, columns=has_dependents.index)
        num_rows = has_dependents.shape[0]
        all_pairs = [(i, j) for i in range(num_rows) for j in range(i + 1, num_rows)]
        num_connections = int(connection_percentage * len(all_pairs))
        selected_pairs = set(np.random.choice(len(all_pairs), size=num_connections, replace=False))
        
        # Build connections
        for idx in selected_pairs:
            i, j = all_pairs[idx]
            adj_matrix.loc[has_dependents.index[i], has_dependents.index[j]] = 1
            adj_matrix.loc[has_dependents.index[j], has_dependents.index[i]] = 1
        
        # Create graph
        G = nx.from_pandas_adjacency(adj_matrix)
        
        # Add missing nodes
        all_node_indices = X_train_subset.index
        existing_node_indices = set(G.nodes)
        missing_node_indices = set(all_node_indices) - existing_node_indices
        G.add_nodes_from(missing_node_indices)
        
        return G
    
    @staticmethod
    def visualize_graph(G, X_train_subset):
        # Sample nodes for visualization
        lcc_nodes = list(max(nx.connected_components(G), key=len))
        lcc_sample = random.sample(lcc_nodes, min(40, len(lcc_nodes)))
        non_lcc_nodes = [n for n in G.nodes if n not in lcc_nodes]
        non_lcc_sample = random.sample(non_lcc_nodes, min(60, len(non_lcc_nodes)))
        sample_nodes = lcc_sample + non_lcc_sample
        subgraph = G.subgraph(sample_nodes)
        
        # Create positions
        pos_lcc = nx.kamada_kawai_layout(subgraph.subgraph(lcc_sample), weight=None)
        pos_non_lcc = {n: (random.uniform(-1, 1), random.uniform(-1, 1)) for n in non_lcc_sample}
        pos_full = {**pos_lcc, **pos_non_lcc}
        
        # Plot
        node_colors = ['skyblue' if n in lcc_sample else 'lightcoral' for n in subgraph.nodes]
        plt.figure(figsize=(10, 8))
        nx.draw(
            subgraph, pos=pos_full,
            with_labels=False, node_size=400,
            node_color=node_colors, edge_color='gray', width=1.0
        )
        plt.title("Graph with Kamada-Kawai Layout (LCC Centered, Red Nodes Spread)")
        plt.show()
