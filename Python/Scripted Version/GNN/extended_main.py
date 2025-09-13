#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core libraries
import random
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Sampling sizes (kept as in your original code)
SAMPLE_SIZES = {'non_default': 7000, 'default': 3000}
TEST_SIZE = 0.2
CATEGORICAL_COLS = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']
NUMERICAL_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                  'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']

# Graph connection parameter (kept but overridden if desired)
FRAC = 1
FRAC = 0.01 * (FRAC)**(5.3)

# Masking rate: proportion of nodes whose labels are hidden from the GNN
MASK_RATE = 0.2

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================
class DataPreprocessor:
    @staticmethod
    def load_and_sample_data(filepath):
        df = pd.read_csv(filepath)

        df_non_default = df[df['Default'] == 0]
        df_default = df[df['Default'] == 1]

        df_non_default_sampled = resample(
            df_non_default, replace=False, n_samples=SAMPLE_SIZES['non_default'], random_state=RANDOM_SEED
        )
        df_default_sampled = resample(
            df_default, replace=False, n_samples=SAMPLE_SIZES['default'], random_state=RANDOM_SEED
        )

        df_small = pd.concat([df_non_default_sampled, df_default_sampled])
        df_small = df_small.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        # drop LoanID if present
        if 'LoanID' in df_small.columns:
            df_small = df_small.drop(columns=['LoanID'])

        return df_small

    @staticmethod
    def preprocess_data(df):
        
        label_encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        # Features / labels
        X = df.drop(columns=['Default'])
        y = df['Default'].astype(int)

        # Standardize numerical columns
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])

        return X_scaled, y, scaler, label_encoders

#====================================================================================================================
# Graph Construction Module
#====================================================================================================================
class GraphBuilder:
    @staticmethod
    def build_graph(X, y, connection_percentage=FRAC):

        combined = X.copy()
        combined['Default'] = y.values

        default_rows = combined[combined['Default'] == 1]
        has_dependents = default_rows[default_rows['HasDependents'] == 1]

        # adjacency among the subset with dependents
        adj_matrix = pd.DataFrame(0, index=has_dependents.index, columns=has_dependents.index)
        num_rows = has_dependents.shape[0]

        # if there are few rows, avoid errors
        if num_rows >= 2:
            all_pairs = [(i, j) for i in range(num_rows) for j in range(i + 1, num_rows)]
            num_connections = max(1, int(connection_percentage * len(all_pairs)))
            # sample without replacement
            selected_pairs = set(np.random.choice(len(all_pairs), size=num_connections, replace=False))

            for idx in selected_pairs:
                i, j = all_pairs[idx]
                adj_matrix.iloc[i, j] = 1
                adj_matrix.iloc[j, i] = 1

        # create networkx graph from adjacency among the dependents subset
        G = nx.from_pandas_adjacency(adj_matrix)

        # ensure all original node indices appear in the graph (as isolated nodes if needed)
        all_node_indices = X.index
        existing_node_indices = set(G.nodes)
        missing_node_indices = set(all_node_indices) - existing_node_indices
        if missing_node_indices:
            G.add_nodes_from(missing_node_indices)

        return G, adj_matrix

    @staticmethod
    def visualize_graph(G, X, max_lcc=40, max_other=60):
        
        plt.figure(figsize=(10, 8))

        if G.number_of_nodes() == 0:
            print("Graph is empty; nothing to visualize.")
            return

        # find largest connected component
        lcc = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
        lcc_nodes = list(lcc)
        lcc_sample = random.sample(lcc_nodes, min(len(lcc_nodes), max_lcc)) if lcc_nodes else []

        other_nodes = [n for n in G.nodes if n not in lcc]
        other_sample = random.sample(other_nodes, min(len(other_nodes), max_other)) if other_nodes else []

        sample_nodes = lcc_sample + other_sample
        subgraph = G.subgraph(sample_nodes)

        # positions
        pos = nx.spring_layout(subgraph, seed=RANDOM_SEED)

        # color by whether in LCC
        node_colors = ["C0" if n in lcc_sample else "C1" for n in subgraph.nodes()]

        nx.draw(subgraph, pos=pos, node_color=node_colors, with_labels=False, node_size=120, edge_color='gray')
        legend_elements = [Patch(facecolor='C0', label='LCC'), Patch(facecolor='C1', label='Other')]
        plt.legend(handles=legend_elements)
        plt.title("Graph (sampled nodes)")
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
    def train_model(model, data, train_mask, epochs=200, print_interval=20, lr=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)

            # compute loss only on nodes with known labels (train_mask==True)
            mask = train_mask.to(device)
            if mask.sum().item() == 0:
                raise ValueError("Train mask has zero True elements; nothing to learn from.")

            loss = criterion(out[mask], data.y[mask])
            loss.backward()
            optimizer.step()

            if (epoch % print_interval) == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    preds = out.argmax(dim=1)
                    acc = (preds[mask] == data.y[mask]).float().mean().item()
                model.train()
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Masked-train Accuracy: {acc:.4f}")

        return model

#====================================================================================================================
# Model evaluation
#====================================================================================================================
class ModelEvaluator:
    @staticmethod
    def evaluate_xgb_on_splits(X, y, train_idx, test_idx, description="XGBoost"):
        # Train on train_idx and evaluate on test_idx
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        print(f"{description} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["No Default", "Default"]))
        try:
            auc = roc_auc_score(y_test, probs)
            print(f"{description} AUC: {auc:.4f}")
        except Exception:
            print("Could not compute AUC (maybe only one class present in y_test?).")

        ModelEvaluator.plot_confusion_matrix(cm, f"Confusion Matrix: {description}")
        return acc, cm

    @staticmethod
    def plot_confusion_matrix(cm, title):
        cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.85, as_cmap=True)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

#====================================================================================================================
# Simple MLP 
#====================================================================================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
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
    def train_and_evaluate(X_train_df, y_train_ser, X_test_df, y_test_ser, input_dim, epochs=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = torch.FloatTensor(X_train_df.values).to(device)
        y_train = torch.LongTensor(y_train_ser.values).to(device)
        X_test = torch.FloatTensor(X_test_df.values).to(device)
        y_test = torch.LongTensor(y_test_ser.values).to(device)

        model = MLPModel(input_dim).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            opt.zero_grad()
            out = model(X_train)
            loss = crit(out, y_train)
            loss.backward()
            opt.step()
            if epoch % 20 == 0:
                with torch.no_grad():
                    preds = out.argmax(dim=1)
                    acc = (preds == y_train).float().mean().item()
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train acc: {acc:.4f}")

        model.eval()
        with torch.no_grad():
            out_test = model(X_test)
            preds = out_test.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(out_test, dim=1)[:, 1].cpu().numpy()
            acc = accuracy_score(y_test.cpu().numpy(), preds)
            cm = confusion_matrix(y_test.cpu().numpy(), preds)
            print(f"MLP Test Accuracy: {acc:.4f}")
            ModelEvaluator.plot_confusion_matrix(cm, "MLP Confusion Matrix")
            try:
                auc = roc_auc_score(y_test.cpu().numpy(), probs)
                print(f"MLP Test AUC: {auc:.4f}")
            except Exception:
                pass
        return acc, cm

#====================================================================================================================
# Main Execution
#====================================================================================================================
if __name__ == "__main__":
    # 1) Load & preprocess the entire dataset
    df = DataPreprocessor.load_and_sample_data("data/Loan_default.csv")
    X_all, y_all, scaler, label_encoders = DataPreprocessor.preprocess_data(df)

    n = len(X_all)
    print(f"Loaded data with {n} nodes")

    # 2) Build graph using the original logic but on the full dataset
    G, adj_matrix = GraphBuilder.build_graph(X_all, y_all)
    GraphBuilder.visualize_graph(G, X_all)

    # 3) Decide which nodes will be withheld (masked) from the GNN during training.
    all_indices = list(X_all.index)
    num_mask = int(MASK_RATE * n)
    masked_indices = set(random.sample(all_indices, num_mask))
    print(f"Masking {num_mask} nodes ({100*MASK_RATE:.1f}%) for GNN training and reserving them as final test set.")

    # 4) Convert graph to torch_geometric data and attach features & labels
    data = from_networkx(G)

    # Ensure node order consistent with X_all index order
    graph_node_order = list(G.nodes)

    data.x = torch.tensor(X_all.loc[graph_node_order].values, dtype=torch.float)
    data.y = torch.tensor(y_all.loc[graph_node_order].values, dtype=torch.long)

    # Build train_mask: True for nodes that the GNN is allowed to use labels for
    node_to_pos = {node: pos for pos, node in enumerate(graph_node_order)}
    mask_array = np.array([False] * len(graph_node_order))
    for node in graph_node_order:
        if node not in masked_indices:
            mask_array[node_to_pos[node]] = True

    train_mask = torch.tensor(mask_array, dtype=torch.bool)

    # 5) Train GraphSAGE using only the unmasked labels
    model = GraphSAGEModel(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)
    trained_model = GraphTrainer.train_model(model, data, train_mask, epochs=200, print_interval=50)

    # 6) Use the trained GNN to produce embeddings for all nodes
    trained_model.eval()
    with torch.no_grad():
        hidden = trained_model.conv1(data.x, data.edge_index)
        hidden = F.relu(hidden)
        embeddings_np = hidden.cpu().numpy()

    original_features_np = data.x.cpu().numpy()
    labels_np = data.y.cpu().numpy()

    # 7) Downstream supervised evaluation: train on non-masked nodes, test on masked nodes
    #    - Baseline: original features only
    #    - GNN-enhanced: original features + embeddings

    # indices in positional order
    pos_indices = np.arange(len(graph_node_order))
    train_pos_idx = [node_to_pos[i] for i in graph_node_order if i not in masked_indices]
    test_pos_idx = [node_to_pos[i] for i in graph_node_order if i in masked_indices]

    # Baseline (original features)
    ModelEvaluator.evaluate_xgb_on_splits(original_features_np, labels_np, train_pos_idx, test_pos_idx, description="XGBoost (original features)")

    # With embeddings
    combined_features = np.hstack((original_features_np, embeddings_np))
    ModelEvaluator.evaluate_xgb_on_splits(combined_features, labels_np, train_pos_idx, test_pos_idx, description="XGBoost (original+GNN embeddings)")

    # Evaluate using the raw adjacency matrix as features for comparison
    try:
        # align adjacency to positional order; adjacency only originally built for "dependents subset",
        # so fill missing rows/cols with zeros for full node set
        adj_full = pd.DataFrame(0, index=graph_node_order, columns=graph_node_order)
        # adj_matrix had index equal to the subset indices that had dependents. Copy those into the full matrix.
        for i in adj_matrix.index:
            for j in adj_matrix.columns:
                if i in adj_full.index and j in adj_full.columns:
                    adj_full.loc[i, j] = adj_matrix.loc[i, j]

        adj_np = adj_full.loc[graph_node_order, graph_node_order].values
        adj_augmented = np.hstack((original_features_np, adj_np))
        ModelEvaluator.evaluate_xgb_on_splits(adj_augmented, labels_np, train_pos_idx, test_pos_idx, description="XGBoost (original+raw-adj)")
    except Exception as e:
        print("Skipping adjacency-based evaluation due to:", e)

    # 8) Quick MLP comparison using the same train/test split (original features only)
    X_df = pd.DataFrame(original_features_np, index=graph_node_order)
    y_ser = pd.Series(labels_np, index=graph_node_order)
    X_train_df = X_df.iloc[train_pos_idx]
    y_train_ser = y_ser.iloc[train_pos_idx]
    X_test_df = X_df.iloc[test_pos_idx]
    y_test_ser = y_ser.iloc[test_pos_idx]

    try:
        MLPTrainer.train_and_evaluate(X_train_df, y_train_ser, X_test_df, y_test_ser, input_dim=X_df.shape[1], epochs=60)
    except Exception as e:
        print("MLP training failed:", e)

# %%
