#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

# Constants
RANDOM_SEED = 42
SAMPLE_SIZES = {'non_default': 7000, 'default': 3000}
TEST_SIZE = 0.2
CATEGORICAL_COLS = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']
NUMERICAL_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                  'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================
class DataPreprocessor:
    @staticmethod
    def load_and_sample_data(filepath):
        df = pd.read_csv(filepath)
        
        # Split by class and sample
        df_non_default = df[df['Default'] == 0]
        df_default = df[df['Default'] == 1]
        
        df_non_default_sampled = resample(df_non_default, replace=False, 
                                        n_samples=SAMPLE_SIZES['non_default'], 
                                        random_state=RANDOM_SEED)
        df_default_sampled = resample(df_default, replace=False, 
                                     n_samples=SAMPLE_SIZES['default'], 
                                     random_state=RANDOM_SEED)
        
        # Combine and shuffle
        df_small = pd.concat([df_non_default_sampled, df_default_sampled])
        df_small = df_small.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        df_small = df_small.drop(columns=['LoanID'])
        
        return df_small
    
    @staticmethod
    def preprocess_data(df):
        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in CATEGORICAL_COLS:
            df[col] = label_encoder.fit_transform(df[col])
        
        # Split features and target
        X = df.drop(columns=["Default"])
        y = df["Default"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        
        # Standardize numerical data
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
        X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
        
        return X_train_scaled, X_test_scaled, y_train, y_test

#====================================================================================================================
# Graph Construction Module
#====================================================================================================================
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

#====================================================================================================================
# GraphSAGE Model Module
#====================================================================================================================
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphTrainer:
    @staticmethod
    def train_model(model, data, epochs=1000, print_interval=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % print_interval == 0:
                pred = out.argmax(dim=1)
                acc = (pred == data.y).sum().item() / data.y.size(0)
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
        
        return model

#====================================================================================================================
# XGBoost Evaluation Module
#====================================================================================================================
class ModelEvaluator:
    @staticmethod
    def evaluate_with_gnn(model, data, original_features_np, labels):
        # Get GNN embeddings
        model.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            hidden_embeddings = model.conv1(x, edge_index)
            hidden_embeddings = F.relu(hidden_embeddings)
            embeddings_np = hidden_embeddings.cpu().numpy()
        
        # Combine features
        combined_features = np.hstack((original_features_np, embeddings_np))
        
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            combined_features, labels, test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, stratify=labels
        )
        
        # Train and evaluate XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"XGBoost Accuracy (with GNN embeddings): {acc:.4f}")
        ModelEvaluator.plot_confusion_matrix(cm, "XGBoost Confusion Matrix (GNN-enhanced features)")
        
        return acc, cm
    
    @staticmethod
    def evaluate_without_gnn(original_features_np, labels):
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            original_features_np, labels, test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, stratify=labels
        )
        
        # Train and evaluate XGBoost
        xgb_raw = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        xgb_raw.fit(X_train, y_train)
        y_pred_raw = xgb_raw.predict(X_val)
        acc_raw = accuracy_score(y_val, y_pred_raw)
        cm_raw = confusion_matrix(y_val, y_pred_raw)
        
        print(f"XGBoost Accuracy (original features only): {acc_raw:.4f}")
        ModelEvaluator.plot_confusion_matrix(cm_raw, "Confusion Matrix: XGBoost (Original Features Only)")
        
        return acc_raw, cm_raw
    
    @staticmethod
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'], 
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

#====================================================================================================================
# Main Execution
#====================================================================================================================
if __name__ == "__main__":
    # 1. Data Preprocessing
    df = DataPreprocessor.load_and_sample_data("data/Loan_default.csv")
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
