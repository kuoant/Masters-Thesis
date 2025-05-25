

#=================================================================================================
#%% Import
#=================================================================================================

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

#=================================================================================================
#%% Artificial Data Generation based on Karate Club Data Set
#=================================================================================================

# Generate the graph with 2 groups
sizes = [50, 50]
p_in, p_out = 0.1, 0.002
probs = [[p_in, p_out], [p_out, p_in]]
G = nx.stochastic_block_model(sizes, probs, seed=42)
A = nx.to_numpy_array(G)
labels = [0]*50 + [1]*50  # Ground-truth group assignments

# Generate random feature DataFrame
np.random.seed(42)

first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Riley", "Jamie", "Casey", "Robin", "Sam", "Drew"]
last_names = ["Smith", "Lee", "Brown", "Wilson", "Johnson", "Clark", "Lewis", "Walker", "Hall", "Allen"]
hobbies_pool = ["chess", "biking", "gaming", "cooking", "reading", "swimming", "photography", "hiking"]
jobs = ["engineer", "teacher", "nurse", "designer", "developer", "analyst", "writer", "plumber", "lawyer", "chef"]

data = []
for _ in range(100):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(18, 65)
    years_in_club = random.randint(0, 10)
    hobbies = ', '.join(random.sample(hobbies_pool, k=random.randint(1, 3)))
    job = random.choice(jobs)
    data.append([name, age, years_in_club, hobbies, job])

df = pd.DataFrame(data, columns=["name", "age", "years_in_club", "hobbies", "job"])

df = np.hstack([df, A])

# Extract structured features
df_structured = pd.DataFrame(data, columns=["name", "age", "years_in_club", "hobbies", "job"])

# Drop 'name' since it's just random noise text (not useful even as a category)
df_structured = df_structured.drop(columns=['name'])

# One-hot encode categorical features
categorical_cols = ['hobbies', 'job']
ohe = OneHotEncoder(sparse_output=False)
cat_encoded = ohe.fit_transform(df_structured[categorical_cols])

# Combine numerical + one-hot categorical features
X_random = np.hstack([
    df_structured[['age', 'years_in_club']].values,
    cat_encoded
])

pd.DataFrame(df)


#=================================================================================================
#%% XGBoost
#=================================================================================================


# Final feature matrix: random features + adjacency matrix
X = np.hstack([X_random, A])
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (XGBoost)")
plt.show()




#=================================================================================================
# %% Plot the Observation
#=================================================================================================


# Color nodes by their group label (0 or 1)
colors = ['skyblue' if label == 0 else 'lightcoral' for label in labels]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=80, edge_color='gray')
plt.title("Two-Group SBM (100 Nodes) with Sparse Inter-Group Connections")
plt.show()





#=================================================================================================
#%% Some Data Preprocessing to make sure 
#=================================================================================================


n_nodes = 100

# Convert numpy arrays to torch tensors
X_train_graph = torch.tensor(X_train[:, -n_nodes:], dtype=torch.float)
X_test_graph = torch.tensor(X_test[:, -n_nodes:], dtype=torch.float)

# Stack them
X_graph = torch.cat([X_train_graph, X_test_graph], dim=0)  # shape (100, 100)

# Stack labels too
y = torch.cat([
    torch.tensor(y_train, dtype=torch.long),
    torch.tensor(y_test, dtype=torch.long)
], dim=0)

# Masks
n_nodes = X_graph.shape[0]
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[:70] = True
test_mask[70:] = True

# PyG data object
data = Data(x=X_graph, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

adj_sparse = csr_matrix(A)

# Convert to edge_index (PyG format)
edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
data.edge_index = edge_index


#=================================================================================================
#%% GraphSAGE
#=================================================================================================


# GraphSAGE model definition
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(data.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 2)  # 2 classes

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(hidden_channels=16).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Test function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# Train the model
for epoch in range(1, 201):
    loss = train()
    if epoch % 50 == 0:
        test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

# Final evaluation
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1).cpu().numpy()
true = data.y.cpu().numpy()

test_pred = pred[data.test_mask.cpu().numpy()]
test_true = true[data.test_mask.cpu().numpy()]

# Calculate metrics
test_acc = accuracy_score(test_true, test_pred)
print(f"\nFinal Test Accuracy: {test_acc:.3f}")

cm = confusion_matrix(test_true, test_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Group 0', 'Group 1'])
disp.plot(cmap='Blues', values_format='d')
plt.title("GraphSAGE Confusion Matrix")
plt.show()



# %%


# Get embeddings
embeddings = model.get_embeddings(data.x, data.edge_index)
embeddings_np = embeddings.cpu().numpy()

# Get labels
labels_np = data.y.cpu().numpy()

# Create train/test split using the masks
train_mask = data.train_mask.cpu().numpy()
test_mask = data.test_mask.cpu().numpy()

X_train, y_train = embeddings_np[train_mask], labels_np[train_mask]
X_test, y_test = embeddings_np[test_mask], labels_np[test_mask]

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Predict
y_pred = xgb.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy on GraphSAGE Embeddings: {acc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Group 0', 'Group 1'])
disp.plot(cmap='Greens', values_format='d')
plt.title("XGBoost on GraphSAGE Embeddings")
plt.show()




# %%
