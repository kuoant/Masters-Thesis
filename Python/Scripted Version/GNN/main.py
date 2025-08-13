#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import random
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from matplotlib.colors import rgb2hex
from matplotlib.patches import Patch

# Network Analysis & Graph Conversion
import networkx as nx
from torch_geometric.utils import from_networkx

# PyTorch & PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv

# Machine Learning & Preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Dimensionality Reduction & Evaluation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)   

# Constants
RANDOM_SEED = 42
SAMPLE_SIZES = {'non_default': 7000, 'default': 3000}
TEST_SIZE = 0.2
CATEGORICAL_COLS = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']
NUMERICAL_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                  'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']

# Parameter for controlling connections
FRAC = 1.0

# Target mapping to get comparable results as with transformers
FRAC = 0.01 * (FRAC)**(5.3)

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
    def build_graph(X_train_subset, y_train_subset, connection_percentage=FRAC):
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
        
        return G, adj_matrix
    
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
        
        # Generate cubehelix palette with reversed dark-to-light tones
        palette = sns.cubehelix_palette(
            start=0.5, rot=-0.5, 
            dark=0.7, light=0.3, 
            n_colors=2, 
            reverse=True
        )
        
        # Assign colors
        node_colors = [palette[0] if n in lcc_sample else palette[1] for n in subgraph.nodes]
        
        # Plot
        plt.figure(figsize=(10, 8))
        nx.draw(
            subgraph, pos=pos_full,
            with_labels=False, node_size=400,
            node_color=node_colors, edge_color='gray', width=1.0
        )
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
        
        print("Classification report (XGBoost with GNN embeddings):")
        print(classification_report(y_val, y_pred, target_names=["No Default", "Default"]))

        # Compute predicted probabilities for positive class
        y_probs = xgb_model.predict_proba(X_val)[:, 1]

        # Compute AUC score
        auc = roc_auc_score(y_val, y_probs)

        print(f"XGBoost AUC (with GNN embeddings): {auc:.4f}")

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
        
        print("Classification report (XGBoost without GNN embeddings):")
        print(classification_report(y_val, y_pred_raw, target_names=["No Default", "Default"]))
        
        # Compute predicted probabilities for positive class
        y_probs_raw = xgb_raw.predict_proba(X_val)[:, 1]

        # Compute AUC score
        auc_raw = roc_auc_score(y_val, y_probs_raw)

        print(f"XGBoost AUC (original features only): {auc_raw:.4f}")

        return acc_raw, cm_raw
    
    @staticmethod
    def evaluate_with_adj(original_features_np, labels, adj_matrix):
        
        # Get the indices that exist in both the features and adjacency matrix
        common_indices = adj_matrix.index.intersection(pd.RangeIndex(start=0, stop=len(original_features_np)))
        
        # Filter both features and adjacency matrix to only include common indices
        original_features_filtered = original_features_np[common_indices]
        adj_features = adj_matrix.loc[common_indices, common_indices].values
        labels_filtered = labels[common_indices]
        
        # Combine with original features
        combined_features = np.hstack((original_features_filtered, adj_features))
        
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            combined_features, labels_filtered, test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, stratify=labels_filtered
        )
        
        # Train and evaluate XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"XGBoost Accuracy (with raw adjacency matrix features): {acc:.4f}")
        ModelEvaluator.plot_confusion_matrix(cm, "XGBoost Confusion Matrix (Raw Adjacency Matrix features)")
        
        print("Classification report (XGBoost with raw adjacency matrix features):")
        print(classification_report(y_val, y_pred, target_names=["No Default", "Default"]))

        # Compute predicted probabilities for positive class
        y_probs = xgb_model.predict_proba(X_val)[:, 1]

        # Compute AUC score
        auc = roc_auc_score(y_val, y_probs)

        print(f"XGBoost AUC (with raw adjacency matrix features): {auc:.4f}")

        return acc, cm
        
    @staticmethod
    def plot_confusion_matrix(cm, title):
        cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.85, as_cmap=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1']
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


#====================================================================================================================
# MLP Model Module
#====================================================================================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MLPTrainer:
    @staticmethod
    def train_and_evaluate(X_train, y_train, X_test, y_test, input_dim, hidden_dim=64, output_dim=2, epochs=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.LongTensor(y_train.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_test_tensor = torch.LongTensor(y_test.values).to(device)
        
        # Initialize model
        model = MLPModel(input_dim, hidden_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == y_train_tensor).sum().item()
                    acc = correct / y_train_tensor.size(0)
                    print(f'Epoch {epoch} | Loss: {loss.item():.4f} | Training Accuracy: {acc:.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            acc = correct / y_test_tensor.size(0)
            cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
            
            print(f'MLP Test Accuracy: {acc:.4f}')
            ModelEvaluator.plot_confusion_matrix(cm, "MLP Confusion Matrix")

            # Classification Report
            print("Classification report (MLP):")
            print(classification_report(
                y_test_tensor.cpu(), predicted.cpu(), 
                target_names=["No Default", "Default"]
            ))

            # Compute probabilities for positive class
            probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()

            auc = roc_auc_score(y_true, probs)
            print(f'MLP Test AUC: {auc:.4f}')
                    
        return acc, cm

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
    G, adj_matrix = GraphBuilder.build_graph(X_train_subset, y_train_subset)
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

    # With adjacency matrix
    ModelEvaluator.evaluate_with_adj(original_features_np, labels, adj_matrix)

    # 6. MLP Performance Comparison
    mlp_acc, mlp_cm = MLPTrainer.train_and_evaluate(
        X_train_scaled, y_train, 
        X_test_scaled, y_test,
        input_dim=X_train_scaled.shape[1]
    )

    # 7. Visualize Embeddings with TSNE
    trained_model.eval()
    with torch.no_grad():
        hidden_embeddings = trained_model.conv1(data.x, data.edge_index)
        hidden_embeddings = F.relu(hidden_embeddings)
        embeddings_np = hidden_embeddings.cpu().numpy()

    original_features_np = data.x.cpu().numpy()
    combined_features = np.hstack((original_features_np, embeddings_np))
    labels_np = data.y.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED)
    emb_2d = tsne.fit_transform(embeddings_np)
    cubehelix_colors = sns.cubehelix_palette(
        start=0.5, rot=-0.5, 
        dark=0.7, light=0.3, 
        n_colors=2, 
        reverse=False
    )
    label_to_color = {0: cubehelix_colors[0], 1: cubehelix_colors[1]}
    point_colors = [label_to_color[label] for label in labels_np]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=point_colors, alpha=0.7, edgecolor='none'
    )
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    
    
    legend_elements = [
        Patch(facecolor=cubehelix_colors[0], label='Class 0'),
        Patch(facecolor=cubehelix_colors[1], label='Class 1')
    ]
    plt.legend(handles=legend_elements, title="Default", loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8. Check Importance of Embeddings
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
    xgb_model.fit(combined_features, labels_np)

    importances = xgb_model.feature_importances_
    feature_names = [f'orig_{i}' for i in range(original_features_np.shape[1])] + [f'emb_{i}' for i in range(embeddings_np.shape[1])]

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_feats = importance_df.sort_values(by='Importance', ascending=False).head(20)

    # Cubehelix color palette (reversed: light to dark)
    cubehelix_palette = sns.cubehelix_palette(
        start=0.5, rot=-0.5, 
        dark=0.3, light=0.8, 
        n_colors=20,
        reverse=True
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_feats, x='Importance', y='Feature', palette=cubehelix_palette)
    plt.tight_layout()
    plt.show()


    # 9. Visualize PCA of 3D-Embeddings
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings_np)

    # Convert labels
    label_str = ['Default' if label == 1 else 'Non-default' for label in labels_np]

    # Sample points for better visualization
    sample_size = 1000
    total_points = len(label_str)

    if total_points > sample_size:
        sampled_indices = random.sample(range(total_points), sample_size)
    else:
        sampled_indices = list(range(total_points))

    embeddings_3d_sampled = embeddings_3d[sampled_indices]
    label_str_sampled = [label_str[i] for i in sampled_indices]

    # Use the same cubehelix colors
    cubehelix_colors = sns.cubehelix_palette(
        start=0.5, rot=-0.5,
        dark=0.7, light=0.3,
        n_colors=2, reverse=True
    )

    # Convert RGB to hex using matplotlib
    default_hex = rgb2hex(cubehelix_colors[0])
    nondefault_hex = rgb2hex(cubehelix_colors[1])

    # Plotly color map
    color_map = {
        'Default': default_hex,
        'Non-default': nondefault_hex
    }

    # Plot
    fig = px.scatter_3d(
        x=embeddings_3d_sampled[:, 0],
        y=embeddings_3d_sampled[:, 1],
        z=embeddings_3d_sampled[:, 2],
        color=label_str_sampled,
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Label'},
        title='3D PCA of GNN Embeddings',
        opacity=1.0,
        color_discrete_map=color_map
    )

    pio.renderers.default = 'browser'
    fig.show()



# %%
