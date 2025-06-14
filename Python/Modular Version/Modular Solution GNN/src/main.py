#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Replace with your path to src
os.chdir("/Users/fabiankuonen/Desktop/Masters Thesis/Python/Modular Version/Modular Solution GNN/src")

from src.data_preprocessing import DataPreprocessor
from src.graph_builder import GraphBuilder
from models.graphsage import GraphSAGEModel
from models.trainer import GraphTrainer
from src.evaluation import ModelEvaluator

import torch
from torch_geometric.utils import from_networkx

if __name__ == "__main__":
    # 1. Data Preprocessing
    df = DataPreprocessor.load_and_sample_data("../data/raw/Loan_default.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = DataPreprocessor.preprocess_data(df)
    
    # 2. Graph Construction
    X_train_subset = X_train_scaled[:10000]
    y_train_subset = y_train[:10000]
    G = GraphBuilder.build_graph(X_train_subset, y_train_subset)
    GraphBuilder.visualize_graph(G, X_train_subset)
    
    # 3. Convert to PyTorch Geometric format
    graph_node_order = list(G.nodes)
    data = from_networkx(G)
    data.x = torch.tensor(X_train_subset.loc[graph_node_order].values, dtype=torch.float)
    data.y = torch.tensor(y_train_subset.loc[graph_node_order].values, dtype=torch.long)
    
    # 4. Train GraphSAGE
    model = GraphSAGEModel(data.num_node_features, 64, 2)
    trained_model = GraphTrainer.train_model(model, data)
    
    # 5. Evaluate with and without GNN features
    original_features_np = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    # With GNN embeddings
    ModelEvaluator.evaluate_with_gnn(trained_model, data, original_features_np, labels)
    
    # Without GNN embeddings (original features only)
    ModelEvaluator.evaluate_without_gnn(original_features_np, labels)

# %%
