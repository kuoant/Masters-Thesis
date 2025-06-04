#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import random

# Seed for Reproduciblity
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and clean data
df = pd.read_csv('housing.csv')
sample_df = df.sample(n=500, random_state=42)
sample_df = sample_df[sample_df["median_house_value"] < 500000].dropna().reset_index(drop=True)

# Prepare graph
def build_housing_graph(df, k=30):
    coords = df[['longitude', 'latitude']].values
    dist_matrix = np.sqrt(((coords[:, None] - coords) ** 2).sum(axis=2))
    neighbors = np.argpartition(dist_matrix, k + 1, axis=1)[:, 1:k+1]

    edge_index = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop('median_house_value', axis=1)
    x = StandardScaler().fit_transform(numeric_features)
    x = torch.tensor(x, dtype=torch.float)

    y = torch.tensor(df['median_house_value'].values, dtype=torch.float)

    # Create train/test masks
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

graph_data = build_housing_graph(sample_df)

# Define GraphSAGE model
class GraphSAGERegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, embedding_dim=16):
        super().__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        out = self.linear(x).squeeze()
        return x, out  # embeddings and prediction

# Train the model
def train_model(data, epochs=5000):
    model = GraphSAGERegressor(input_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.2f}")
    return model

model = train_model(graph_data)

# Evaluate on test set
model.eval()
with torch.no_grad():
    embeddings, predictions = model(graph_data.x, graph_data.edge_index)
    y_true = graph_data.y[graph_data.test_mask].numpy()
    y_pred = predictions[graph_data.test_mask].numpy()

    print("\nGraphSAGE Regressor (Masked) Results:")
    print(f"R² score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('GraphSAGE Node Regression (No Data Leakage)')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.grid(True)
    plt.show()


# %%

# Linear Regression on training nodes and evaluation on test nodes
X_train = graph_data.x[graph_data.train_mask].numpy()
y_train = graph_data.y[graph_data.train_mask].numpy()
X_test = graph_data.x[graph_data.test_mask].numpy()
y_test = graph_data.y[graph_data.test_mask].numpy()

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)

print("\nLinear Regression Results:")
print(f"R² score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")


# %%

# Prepare combined features: original + embeddings
model.eval()
with torch.no_grad():
    embeddings, _ = model(graph_data.x, graph_data.edge_index)

X_orig = graph_data.x.numpy()
X_combined = np.hstack([X_orig, embeddings.numpy()])

X_train_combined = X_combined[graph_data.train_mask.numpy()]
y_train = graph_data.y[graph_data.train_mask].numpy()
X_test_combined = X_combined[graph_data.test_mask.numpy()]
y_test = graph_data.y[graph_data.test_mask].numpy()

linreg_combined = LinearRegression()
linreg_combined.fit(X_train_combined, y_train)
y_pred_combined = linreg_combined.predict(X_test_combined)

print("\nLinear Regression on Combined Features (Original + Embeddings):")
print(f"R² score: {r2_score(y_test, y_pred_combined):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_combined)):.2f}")

# %%
