#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import random
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.io as pio

# PyTorch & PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Machine Learning & Preprocessing
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Decomposition
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

# Graph & Location
import geopandas as gpd
import contextily as ctx
import networkx as nx
from shapely.geometry import Point, LineString

# Seed for Reproduciblity
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================

# Load and clean data
df = pd.read_csv('data/housing.csv')
sample_df = df.sample(n=500, random_state=42)
sample_df = sample_df[sample_df["median_house_value"] < 500000].dropna().reset_index(drop=True)

#====================================================================================================================
# Graph Construction Module
#====================================================================================================================

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

#====================================================================================================================
# GraphSAGE Model Module
#====================================================================================================================

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

    palette = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, n_colors=2, reverse=True)
    point_color = palette[0]
    line_color = palette[1]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k', color=point_color)
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('GraphSAGE Node Regression')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color=line_color)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#====================================================================================================================
# Linear Regression Model Module
#====================================================================================================================

# Linear Regression on training nodes and evaluation on test nodes
X_train = graph_data.x[graph_data.train_mask].numpy()
y_train = graph_data.y[graph_data.train_mask].numpy()
X_test = graph_data.x[graph_data.test_mask].numpy()
y_test = graph_data.y[graph_data.test_mask].numpy()

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)

print("\nLinear Regression Results:")
print(f"R² score on Test Set: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE on Test Set: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")

#====================================================================================================================
# Linear Regression + Embeddings Model Module
#====================================================================================================================

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
print(f"R² score on Test Set: {r2_score(y_test, y_pred_combined):.4f}")
print(f"RMSE on Test Set: {np.sqrt(mean_squared_error(y_test, y_pred_combined)):.2f}")

#====================================================================================================================
# Embeddings Visualization
#====================================================================================================================

# 1. Clustering
embeddings_np = embeddings.numpy()

inertia = []
K_range = range(1, 15) 

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings_np)
    inertia.append(kmeans.inertia_)

cubehelix_colors = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, n_colors=2, reverse=True)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', color=cubehelix_colors[1], markerfacecolor=cubehelix_colors[0], markeredgecolor=cubehelix_colors[1])
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Plot for KMeans Clustering on Embeddings')
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 3D PCA 
embeddings_np = embeddings.numpy()
targets = graph_data.y.numpy()

# PCA to reduce embeddings to 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings_np)

cubehelix_cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, as_cmap=True)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
    c=targets, cmap=cubehelix_cmap, alpha=0.7
)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
fig.colorbar(sc, ax=ax, label='Median House Value')
plt.show()

# Interactive 3D-PCA Plot
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings_np)

cubehelix_colors = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, n_colors=256, as_cmap=False)
cubehelix_colors_hex = [mcolors.rgb2hex(c) for c in cubehelix_colors]

fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    color=targets,
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
    title='Interactive 3D PCA of GraphSAGE Embeddings',
    opacity=0.7,
    color_continuous_scale=cubehelix_colors_hex
)

pio.renderers.default = 'browser'
fig.show()

# 3. Linear Model Coefficients
coefficients = linreg_combined.coef_
n_features = graph_data.x.shape[1]
embedding_coefs = coefficients[n_features:] 

n_bars = len(embedding_coefs)
palette = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, n_colors=n_bars)

plt.figure(figsize=(8, 4))
sns.barplot(
    x=np.arange(n_bars),
    y=embedding_coefs,
    palette=palette
)
plt.xlabel("Embedding Dimension")
plt.ylabel("Coefficient")
plt.title("Linear Model Coefficients for GNN Embeddings")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Permutation Importance of Embeddings
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_combined))
importances = []

for i in range(embeddings.shape[1]):
    X_temp = X_test_combined.copy()
    np.random.shuffle(X_temp[:, graph_data.x.shape[1] + i])  # shuffle one embedding dim
    y_temp_pred = linreg_combined.predict(X_temp)
    rmse = np.sqrt(mean_squared_error(y_test, y_temp_pred))
    importances.append(rmse - baseline_rmse)

n_bars = len(importances)
palette = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, n_colors=n_bars)

plt.figure(figsize=(8, 4))
sns.barplot(x=np.arange(n_bars), y=importances, palette=palette)
plt.xlabel("Embedding Dimension")
plt.ylabel("RMSE Increase")
plt.title("Permutation Importance of Embedding Dimensions")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Canonical Correlation
cca = CCA(n_components=3)
X_orig_std = StandardScaler().fit_transform(graph_data.x.numpy())
Y_embed_std = StandardScaler().fit_transform(embeddings.numpy())

cca.fit(X_orig_std, Y_embed_std)
X_c, Y_c = cca.transform(X_orig_std, Y_embed_std)

corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(3)]
print("Canonical correlations:", corrs)

#====================================================================================================================
# Regression Outputs
#====================================================================================================================

# For original features
orig_feature_names = graph_data.x.shape[1]
orig_names = sample_df.select_dtypes(include=['int64', 'float64']).drop('median_house_value', axis=1).columns.tolist()

X_train_df = pd.DataFrame(X_train, columns=orig_names)
X_train_sm = sm.add_constant(X_train_df)
model_orig = sm.OLS(y_train, X_train_sm).fit()

print("\nLinear Regression Summary (Original Features):")
print(model_orig.summary())

# For combined features (original + embeddings)
embedding_dim = embeddings.shape[1]
embedding_names = [f"Embedding {i+1}" for i in range(embedding_dim)]

combined_feature_names = orig_names + embedding_names
X_train_combined_df = pd.DataFrame(X_train_combined, columns=combined_feature_names)
X_train_comb_sm = sm.add_constant(X_train_combined_df)
model_combined = sm.OLS(y_train, X_train_comb_sm).fit()

print("\nLinear Regression Summary (Combined Features):")
print(model_combined.summary())

# Run joint F-Test on the Embeddings 
n_total = len(model_combined.params)
n_embedding = len(embedding_names)

R = np.zeros((n_embedding, n_total))

start = n_total - n_embedding
for i in range(n_embedding):
    R[i, start + i] = 1

f_test_result = model_combined.f_test(R)

print("\nF-test for Joint Significance of All Embedding Coefficients:")
print(f_test_result)

#====================================================================================================================
# Linear Regression on PC1
#====================================================================================================================

# Get embeddings from trained model
model.eval()
with torch.no_grad():
    embeddings, _ = model(graph_data.x, graph_data.edge_index)
    emb_np = embeddings.numpy()

# PCA with 1 component (PC1 only)
pca = PCA(n_components=1)
emb_pca = pca.fit_transform(emb_np)

# Split train/test
X_train_pca = emb_pca[graph_data.train_mask.numpy()]
X_test_pca = emb_pca[graph_data.test_mask.numpy()]
y_train = graph_data.y[graph_data.train_mask].numpy()
y_test = graph_data.y[graph_data.test_mask].numpy()

# Add intercept for statsmodels OLS
X_train_pca_sm = sm.add_constant(X_train_pca)
model_pca = sm.OLS(y_train, X_train_pca_sm).fit()

# Predict on test set
X_test_pca_sm = sm.add_constant(X_test_pca)
y_pred_pca = model_pca.predict(X_test_pca_sm)

# Output summary & metrics
print("\nLinear Regression Summary (Using PC1 from embeddings):")
print(model_pca.summary())
print(f"\nR² score on Test Set: {r2_score(y_test, y_pred_pca):.4f}")
print(f"RMSE on Test Set: {np.sqrt(mean_squared_error(y_test, y_pred_pca)):.2f}")

# Extract PC1 and target values
pc1 = emb_pca.squeeze()
y_true = graph_data.y.numpy()

# Compute regression line using the trained statsmodels model
x_vals = np.linspace(pc1.min(), pc1.max(), 100)
x_vals_const = sm.add_constant(x_vals) 
y_vals = model_pca.predict(x_vals_const)

# Plot
color_class_0 = "#5D3A9B"  
color_class_1 = "#3FBFC2"  

plt.figure(figsize=(10, 6))
plt.scatter(pc1, y_true, alpha=0.6, edgecolor='k', color=color_class_0, label='Data')
plt.plot(x_vals, y_vals, color=color_class_1, linewidth=2, label='Regression Line')
plt.xlabel('PC1 from Embeddings')
plt.ylabel('Median House Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#====================================================================================================================
# Graph Visualization
#====================================================================================================================

# Extract coordinates
coords = sample_df[['longitude', 'latitude']].values

# Create GeoDataFrame for nodes
geometry = [Point(xy) for xy in coords]
gdf_nodes = gpd.GeoDataFrame(sample_df, geometry=geometry, crs="EPSG:4326")
gdf_nodes = gdf_nodes.to_crs(epsg=3857)  # Web Mercator for plotting on basemaps

# Build edges without self-loops
edge_index = graph_data.edge_index.numpy()
edges = [(i, j) for i, j in edge_index.T if i != j]

# Create GeoDataFrame for edges
edge_lines = []
for i, j in edges:
    point_i = gdf_nodes.geometry.iloc[i]
    point_j = gdf_nodes.geometry.iloc[j]
    edge_lines.append(LineString([point_i, point_j]))
gdf_edges = gpd.GeoDataFrame(geometry=edge_lines, crs=gdf_nodes.crs)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot edges first so they appear under the nodes
gdf_edges.plot(ax=ax, linewidth=0.3, color='gray', alpha=0.5)

# Plot nodes
gdf_nodes.plot(ax=ax, markersize=15, color='#3FBFC2', alpha=0.8)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

ax.set_axis_off()
plt.show()


# %%
