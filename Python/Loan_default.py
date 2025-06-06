
#====================================================================================================================
#====================================================================================================================
# Import Libraries
#====================================================================================================================
#====================================================================================================================
#%% 

import random
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import xgboost as xgb
import shap

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TextVectorization, GlobalAveragePooling1D



#====================================================================================================================
#====================================================================================================================
# Preprocess Data
#====================================================================================================================
#====================================================================================================================
#%%


# Load data
df = pd.read_csv("Loan_default.csv")

# Split by class
df_non_default = df[df['Default'] == 0]
df_default = df[df['Default'] == 1]

# Sample 7,000 non-defaults and 3,000 defaults
df_non_default_sampled = resample(df_non_default, replace=False, n_samples=7000, random_state=42)
df_default_sampled = resample(df_default, replace=False, n_samples=3000, random_state=42)

# Combine
df_small = pd.concat([df_non_default_sampled, df_default_sampled])

# Shuffle
df = df_small.sample(frac=1, random_state=42).reset_index(drop=True)
df_small = df_small.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.drop(columns=['LoanID'])
df_small = df_small.drop(columns=['LoanID'])

df_small.to_csv("df_small_sampled.csv", index=False)

# Check resultss
print(df['Default'].value_counts())
print(f"Total samples: {len(df)}")

# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split features and target variables
X = df.drop(columns=["Default"])    # Features excluding target
y = df["Default"]                   # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numerical_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'DTIRatio']

X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])


#====================================================================================================================
#====================================================================================================================
# Graph
#====================================================================================================================
#====================================================================================================================
#%%

X_train_subset = X_train_scaled[:10000]
y_train_subset = y_train[:10000]

combined = X_train_subset.copy()
combined['Default'] = y_train_subset.values

default_rows = combined[combined['Default'] == 1]

# Filter rows where HasDependents == 1
has_dependents = default_rows[default_rows['HasDependents'] == 1]

# Set a random seed for reproducibility
np.random.seed(42)

# Number of rows with HasDependents == 1
num_rows = has_dependents.shape[0]

# Initialize an empty adjacency matrix (binary, 1 for connection, 0 for no connection)
adj_matrix = pd.DataFrame(0, index=has_dependents.index, columns=has_dependents.index)

# Generate all possible pairs (i, j) where i < j = (131 nCr 2)
all_pairs = [(i, j) for i in range(num_rows) for j in range(i + 1, num_rows)]


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Calculate x% of the possible connections
num_connections = int(0.05 * len(all_pairs))               
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Randomly select 80% of the pairs
selected_pairs = set(np.random.choice(len(all_pairs), size=num_connections, replace=False))

# Set entries to 1 for the selected pairs using the correct index
for idx in selected_pairs:
    i, j = all_pairs[idx]
    # Use loc to access the rows and columns by their actual index labels
    adj_matrix.loc[has_dependents.index[i], has_dependents.index[j]] = 1
    adj_matrix.loc[has_dependents.index[j], has_dependents.index[i]] = 1  # Undirected

# Create a graph from the adjacency matrix
G = nx.from_pandas_adjacency(adj_matrix)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')

# Show the plot
plt.show()

# Get all node indices from your data
all_node_indices = X_train_subset.index

# Get current node indices already in the graph
existing_node_indices = set(G.nodes)

# Determine which nodes are missing
missing_node_indices = set(all_node_indices) - existing_node_indices

# Add missing nodes to the graph (unconnected)
G.add_nodes_from(missing_node_indices)


# Step 1: Largest Connected Component (LCC) center
lcc_nodes = list(max(nx.connected_components(G), key=len))
lcc_sample = random.sample(lcc_nodes, min(40, len(lcc_nodes)))

# Step 2: Sample other nodes (isolated or small components)
non_lcc_nodes = [n for n in G.nodes if n not in lcc_nodes]
non_lcc_sample = random.sample(non_lcc_nodes, min(60, len(non_lcc_nodes)))

# Combine nodes for final subgraph
sample_nodes = lcc_sample + non_lcc_sample
subgraph = G.subgraph(sample_nodes)

# Step 3: Kamada-Kawai Layout for LCC (Blue Nodes)
pos_lcc = nx.kamada_kawai_layout(subgraph.subgraph(lcc_sample), weight=None)

# Step 4: Assign random positions for the red nodes (isolated)
pos_non_lcc = {n: (random.uniform(-1, 1), random.uniform(-1, 1)) for n in non_lcc_sample}

# Step 5: Combine positions: LCC positions + random for red nodes
pos_full = {**pos_lcc, **pos_non_lcc}

# Colors: blue = connected, red = isolated
node_colors = ['skyblue' if n in lcc_sample else 'lightcoral' for n in subgraph.nodes]

# Plot with natural layout and separate positions
plt.figure(figsize=(10, 8))
nx.draw(
    subgraph, pos=pos_full,
    with_labels=False, node_size=400,
    node_color=node_colors, edge_color='gray', width=1.0
)
plt.title("Graph with Kamada-Kawai Layout (LCC Centered, Red Nodes Spread)")
plt.show()



#====================================================================================================================
#====================================================================================================================
# GraphSAGE
#====================================================================================================================
#====================================================================================================================
#%% 


# Convert the graph to PyTorch Geometric format
graph_node_order = list(G.nodes)  # Node order used in from_networkx
data = from_networkx(G)
data.x = torch.tensor(X_train_subset.loc[graph_node_order].values, dtype=torch.float)
data.y = torch.tensor(y_train_subset.loc[graph_node_order].values, dtype=torch.long)

# Define the GraphSAGE model
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

# Initialize the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGEModel(data.num_node_features, 64, 2).to(device)
data = data.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)  # Forward pass
    loss = criterion(out, data.y)  # Calculate loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        # Accuracy calculation
        pred = out.argmax(dim=1)  # Get predicted labels
        acc = (pred == data.y).sum().item() / data.y.size(0)
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")



#%%

# === Step 1: Get GNN embeddings from the hidden layer ===
model.eval()
with torch.no_grad():
    x, edge_index = data.x, data.edge_index
    hidden_embeddings = model.conv1(x, edge_index)
    hidden_embeddings = F.relu(hidden_embeddings)

# Convert embeddings to numpy
embeddings_np = hidden_embeddings.cpu().numpy()

# === Step 2: Combine original features and GNN embeddings ===
original_features_np = data.x.cpu().numpy()
combined_features = np.hstack((original_features_np, embeddings_np))

# === Step 3: Get labels ===
labels = data.y.cpu().numpy()

# === Step 4: Train/test split (optional but recommended) ===
X_train_combined, X_val_combined, y_train_combined, y_val_combined = train_test_split(
    combined_features, labels, test_size=0.2, random_state=42, stratify=labels
)

# === Step 5: Train XGBoost ===
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_combined, y_train_combined)

# === Step 6: Evaluate ===
y_pred = xgb_model.predict(X_val_combined)
acc = accuracy_score(y_val_combined, y_pred)
cm = confusion_matrix(y_val_combined, y_pred)

print(f"XGBoost Accuracy (with GNN embeddings): {acc:.4f}")

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("XGBoost Confusion Matrix (GNN-enhanced features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


#%%

# === Step: Train XGBoost using ONLY original features ===
xgb_raw = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_raw.fit(X_train_combined[:, :original_features_np.shape[1]], y_train_combined)

# === Step: Evaluate on the same test set ===
y_pred_raw = xgb_raw.predict(X_val_combined[:, :original_features_np.shape[1]])
acc_raw = accuracy_score(y_val_combined, y_pred_raw)
cm_raw = confusion_matrix(y_val_combined, y_pred_raw)

print(f"XGBoost Accuracy (original features only, same test set): {acc_raw:.4f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix: XGBoost (Original Features Only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()






#====================================================================================================================
#====================================================================================================================
# Data Preprocessing for TabTransformer
#====================================================================================================================
#====================================================================================================================
#%% 

df = df_small.copy()

# Define risky job descriptions for hotel/hospitality under 24 months
risky_descriptions = [
    "Worked part-time at a hotel assisting with guest services for 12 months.",
    "Employed part-time in hospitality, primarily at a local hotel front desk for 20 months.",
    "Worked evenings part-time at a hotel restaurant as a server for 10 months."
]

# Define random generic job descriptions
generic_descriptions = [
    "Software engineer in a fintech startup. Developed APIs and maintained backend services.",
    "Teacher at a public high school. Responsible for curriculum planning and grading.",
    "Office administrator managing schedules, invoices, and office supplies.",
    "Sales associate at a retail clothing store providing customer support.",
    "Customer service representative at a call center handling billing inquiries.",
    "Freelance content writer producing marketing materials for small businesses.",
    "Warehouse worker managing inventory and handling logistics support.",
    "Data analyst interpreting sales data and creating performance dashboards."
]

# Create a new column
df['JobDescription'] = None


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Define a risky pool
risky_pool = df[(df['Default'] == 1) & (df['HasDependents'] == 'Yes')]

# Randomly sample x% of those rows
risky_sample = risky_pool.sample(frac=1, random_state=42)  # set seed for reproducibility

# Get the indices
risky_indices = risky_sample.index
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# Assign risky descriptions to all qualifying rows (not just 3)
for i, idx in enumerate(risky_indices):
    df.at[idx, 'JobDescription'] = risky_descriptions[i % len(risky_descriptions)]

# Assign random generic descriptions to all others (including remaining risky default cases)
remaining_indices = df[df['JobDescription'].isna()].index
df.loc[remaining_indices, 'JobDescription'] = np.random.choice(generic_descriptions, size=len(remaining_indices))


risky_keywords = ['hotel', 'hospitality', 'restaurant', 'server']
df['is_risky_job'] = df['JobDescription'].str.lower().apply(
    lambda x: any(kw in x for kw in risky_keywords)
)
print(f"✅ Count of rows with risky job description: {df['is_risky_job'].sum()}")
df = df.drop('is_risky_job', axis=1)

default_count = df['Default'].sum()
print(f"✅ Total number of default cases (Default == 1): {default_count}")



#%%

# ------------------------------
# STEP 1: Split Data
# ------------------------------

# Split into train/test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save raw text for JobDescription before encoding
train_data_raw = train_data.copy()
test_data_raw = test_data.copy()

# Separate labels
y_train = train_data['Default']
y_test = test_data['Default']

# ------------------------------
# STEP 2: Define Columns
# ------------------------------

# Categorical (structured)
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus',
                       'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Numerical
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in categorical_columns + ['Default']]

# ------------------------------
# STEP 3: Encode Categorical Columns
# ------------------------------

# Apply category encoding (integer codes)
for col in categorical_columns:
    train_data[col] = train_data[col].astype('category').cat.codes
    test_data[col] = test_data[col].astype('category').cat.codes

# Collect cardinalities
cat_cardinalities = [df[col].nunique() for col in categorical_columns]
cat_features_info = [(cardinality, 32) for cardinality in cat_cardinalities]  # for embedding layers

# ------------------------------
# STEP 4: Normalize Numerical Columns
# ------------------------------

scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# ------------------------------
# STEP 5: Text Vectorization for JobDescription
# ------------------------------

max_tokens = 1000
output_sequence_length = 20

text_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)

job_descriptions = train_data_raw['JobDescription'].fillna('').astype(str).values
text_vectorizer.adapt(job_descriptions)

# ------------------------------
# STEP 6: Prepare Final Inputs for Keras
# ------------------------------

# Convert inputs to dictionaries for multi-input model
X_train_inputs = {
    "categorical_inputs": train_data[categorical_columns].values,
    "numerical_inputs": train_data[numerical_columns].values,
    "JobDescription": train_data_raw['JobDescription'].values
}

X_test_inputs = {
    "categorical_inputs": test_data[categorical_columns].values,
    "numerical_inputs": test_data[numerical_columns].values,
    "JobDescription": test_data_raw['JobDescription'].values
}

# Final shapes
print(f"✅ X_train categorical: {X_train_inputs['categorical_inputs'].shape}")
print(f"✅ X_train numerical: {X_train_inputs['numerical_inputs'].shape}")
print(f"✅ X_train JobDescription: {X_train_inputs['JobDescription'].shape}")
print(f"✅ y_train: {y_train.shape}")





#====================================================================================================================
#====================================================================================================================
# TabTransformer
#====================================================================================================================
#====================================================================================================================
#%% 

# --- Text vectorizer setup ---
max_tokens = 1000
output_sequence_length = 20

text_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)
text_vectorizer.adapt(train_data_raw['JobDescription'].fillna('').astype(str).values)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(embed_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


# --- Model definition ---
class FullTabTransformer(Model):
    def __init__(self, cat_features_info, num_numerical, embed_dim=32, num_heads=2, num_transformer_blocks=2, text_max_tokens=1000, text_output_len=20):
        super(FullTabTransformer, self).__init__()
        
        # Categorical embeddings + transformer
        self.embeddings = [Embedding(input_dim=card, output_dim=embed_dim) for card, _ in cat_features_info]
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_transformer_blocks)]
        self.flatten = Flatten()
        
        # Numerical
        self.numerical_dense = Dense(32, activation="relu")

        # Text processing
        self.text_embedding = Embedding(input_dim=text_max_tokens, output_dim=32)
        self.text_transformer = TransformerBlock(embed_dim=32, num_heads=2)
        self.text_pooling = GlobalAveragePooling1D()

        # Final MLP: Split for feature extraction
        self.feature_dense = keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu')  # Final embedding before sigmoid
        ])
        self.output_dense = Dense(1, activation='sigmoid')
    
    def call(self, inputs, return_embedding=False):
        cat_inputs, num_inputs, text_inputs = inputs

        # Categorical
        x_cat = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = tf.stack(x_cat, axis=1)
        for transformer in self.transformer_blocks:
            x_cat = transformer(x_cat)
        x_cat = self.flatten(x_cat)

        # Numerical
        x_num = self.numerical_dense(num_inputs)

        # Text
        x_text = self.text_embedding(text_inputs)
        x_text = self.text_transformer(x_text)
        x_text = self.text_pooling(x_text)

        # Combine
        x = tf.concat([x_cat, x_num, x_text], axis=1)
        x = self.feature_dense(x)

        if return_embedding:
            return x  # return the 32-dim embeddings
        return self.output_dense(x)


# --- Preprocessing ---
# Adapt text vectorizer FIRST
text_vectorizer.adapt(train_data_raw['JobDescription'].fillna('').astype(str).values)

# Process text
X_train_text_seq = text_vectorizer(train_data_raw['JobDescription'].fillna('').astype(str).values)
X_test_text_seq = text_vectorizer(test_data_raw['JobDescription'].fillna('').astype(str).values)

# Scale numerical features
scaler = StandardScaler()
num_data_train = scaler.fit_transform(X_train[numerical_columns].values)
num_data_test = scaler.transform(X_test[numerical_columns].values)

# Convert to tensors
cat_data_train = tf.convert_to_tensor(X_train[categorical_columns].values, dtype=tf.int32)
num_data_train = tf.convert_to_tensor(num_data_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)

cat_data_test = tf.convert_to_tensor(X_test[categorical_columns].values, dtype=tf.int32)
num_data_test = tf.convert_to_tensor(num_data_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# --- Training ---
model = FullTabTransformer(cat_features_info=cat_features_info, 
                          num_numerical=len(numerical_columns))

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

history = model.fit(
    (cat_data_train, num_data_train, X_train_text_seq),
    y_train_tensor,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)


# --- 1. Predict Probabilities ---
y_pred_proba = model.predict((cat_data_test, num_data_test, X_test_text_seq))

# --- 2. Convert to Binary Labels ---
y_pred = (y_pred_proba.flatten() > 0.5).astype(int)

# --- 3. Evaluate Metrics ---
acc = accuracy_score(y_test_tensor, y_pred)
prec = precision_score(y_test_tensor, y_pred)
rec = recall_score(y_test_tensor, y_pred)
f1 = f1_score(y_test_tensor, y_pred)
auc = roc_auc_score(y_test_tensor, y_pred_proba)

print("✅ Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test_tensor, y_pred))
print("\nClassification Report:\n", classification_report(y_test_tensor, y_pred))

# --- 4. Plot ROC Curve ---
fpr, tpr, _ = roc_curve(y_test_tensor, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkblue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%%

# Embeddings
X_train_embed = model((cat_data_train, num_data_train, X_train_text_seq), return_embedding=True).numpy()
X_test_embed = model((cat_data_test, num_data_test, X_test_text_seq), return_embedding=True).numpy()

# XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train_embed, y_train.values)

# Predictions and accuracy
y_pred_xgb = xgb_clf.predict(X_test_embed)
acc = accuracy_score(y_test.values, y_pred_xgb)
cm = confusion_matrix(y_test.values, y_pred_xgb)

print(f"✅ XGBoost Accuracy on Transformer Embeddings: {acc:.4f}")

# Plot confusion matrix in same style
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix: XGBoost on Transformer Embeddings")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



#%%


# Combine features
X_train_raw = np.hstack([
    cat_data_train.numpy(),  # categorical int-encoded
    num_data_train.numpy()   # scaled numerical
])

X_test_raw = np.hstack([
    cat_data_test.numpy(),
    num_data_test.numpy()
])

# === Evaluate predictions ===
y_pred_raw = xgb_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test.values, y_pred_raw)
cm_raw = confusion_matrix(y_test.values, y_pred_raw)

print(f"✅ XGBoost Accuracy on Raw Features: {acc_raw:.4f}")

# === Plot confusion matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix: XGBoost on Raw Features")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()





#====================================================================================================================
#====================================================================================================================
# Artificial Neural Network (MLP)
#====================================================================================================================
#====================================================================================================================
#%%

from sklearn.neural_network import MLPClassifier

# Initialize the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 30), max_iter=300, random_state=42)

# Train the MLP model
mlp.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]  # Probability for class 1

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred_mlp = (y_pred_prob_mlp > 0.5).astype(int)

# Compute confusion matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('MLP Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred_mlp))


# %%
