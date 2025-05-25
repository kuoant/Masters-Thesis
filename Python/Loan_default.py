#%% Imports   (https://www.kaggle.com/datasets/ruslankl/loan-default-prediction)

import random
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report





#%% Import & Preprocess Data

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
df = df.drop(columns=['LoanID'])

# Check results
print(df['Default'].value_counts())
print(f"Total samples: {len(df)}")

# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split features and target variables
X = df.drop(columns=["Default"])    # Features (excluding target)
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







#%% XGBoost

import xgboost as xgb
import shap

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define parameters and train the model
params = {
    "objective": "binary:logistic", 
    "eval_metric": "logloss",
    "use_label_encoder": False
}

bst = xgb.train(params, dtrain, num_boost_round=100)

# Prediction
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int) 

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))


#%% Shape

# Create an explainer using the trained booster and the training data
explainer = shap.Explainer(bst)

# Compute SHAP values for the test set
shap_values = explainer(dtest)

# Plot summary plot (beeswarm) of SHAP values
shap.summary_plot(shap_values, features=X_test_scaled, feature_names=X_test_scaled.columns)

# Plot SHAP values for a single instance (e.g., the first test sample)
sample_obs = 0
shap.plots.waterfall(shap_values[sample_obs])
print(X_test.iloc[sample_obs], "\nLabel: ", y_test[sample_obs])

# XGBoost built-in feature importance plot
ax = xgb.plot_importance(bst, max_num_features=10, importance_type='gain', xlabel='Gain')
plt.ylabel("")
plt.title('XGBoost Feature Importance')

for txt in ax.texts:
    txt.set_visible(False)

plt.show()

# Extract SHAP values array
shap_vals_array = shap_values.values

# Indices
wrong_idx = np.where(y_pred != y_test)[0]
correct_idx = np.where(y_pred == y_test)[0]

# Mean abs SHAP for wrong and correct
mean_abs_wrong = np.abs(shap_vals_array[wrong_idx]).mean(axis=0)
mean_abs_correct = np.abs(shap_vals_array[correct_idx]).mean(axis=0)

# Difference
diff = mean_abs_wrong - mean_abs_correct

feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(diff.shape[0])]
diff_series = pd.Series(diff, index=feature_names).sort_values(ascending=False)

print("Features with higher impact on wrong predictions (compared to correct):")
print(diff_series.head(10))

# Plot
diff_series.plot(kind='barh')
plt.xlabel('Mean |SHAP value| difference (wrong - correct)')
plt.title('Features Driving Wrong Predictions Differently')
plt.gca().invert_yaxis()
plt.show()


#%%

from scipy.stats import ks_2samp

# Use iloc for positional indexing
interest_wrong = X_test.iloc[wrong_idx]['InterestRate']
interest_correct = X_test.iloc[correct_idx]['InterestRate']

# Plot distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(interest_correct, label='Correct Predictions', shade=True)
sns.kdeplot(interest_wrong, label='Wrong Predictions', shade=True)
plt.title('InterestRate Distribution: Correct vs Wrong Predictions')
plt.xlabel('InterestRate')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Statistical test for distribution difference
stat, p_value = ks_2samp(interest_correct, interest_wrong)
print(f"KS-test statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("The distributions of InterestRate between wrong and correct predictions are significantly different.")
else:
    print("No significant difference detected between the InterestRate distributions for wrong and correct predictions.")













# %% Isolation Forest

from sklearn.ensemble import IsolationForest

# Train Isolation Forest model
iso_forest = IsolationForest(contamination=0.3, random_state=42)  # contamination should be roughly the fraction of outliers
iso_forest.fit(X_train_scaled)

# Prediction
y_pred = iso_forest.predict(X_test_scaled)

# Mapping
y_pred = (y_pred == -1).astype(int)  # Outliers (-1) are predicted as "default" (1), and inliers (1) as "not default" (0)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))






















#%%  Bayesian Logistic Regression Using NUTS

import pymc as pm

X_train_subset = X_train_scaled[:8000]
y_train_subset = y_train[:8000]


with pm.Model() as bayesian_model:
    # Priors on parameters (use moderately informative priors)
    beta_0              = pm.Normal('beta_0', mu=0, sigma=10)
    beta_age            = pm.Normal('beta_age', mu=0, sigma=10)
    beta_income         = pm.Normal('beta_income', mu=0, sigma=10)
    beta_loanAmount     = pm.Normal('beta_loanAmount', mu=0, sigma=10)
    beta_creditScore    = pm.Normal('beta_creditScore', mu=0, sigma=10)
    # beta_monthsEmployed = pm.Normal('beta_monthsEmployed', mu=0, sigma=10)
    # beta_numCreditLines = pm.Normal('beta_numCreditLines', mu=0, sigma=10)
    beta_interestRate   = pm.Normal('beta_interestRate', mu=0, sigma=10)
    # beta_loanTerm       = pm.Normal('beta_loanTerm', mu=0, sigma=10)
    # beta_DTIRatio       = pm.Normal('beta_DTIRatio', mu=0, sigma=10)
    # beta_Education      = pm.Normal('beta_Education', mu=0, sigma=10)
    # beta_EmploymentType = pm.Normal('beta_EmploymentType', mu=0, sigma=10)
    # beta_MaritalStatus  = pm.Normal('beta_MaritalStatus', mu=0, sigma=10)
    # beta_HasMortgage    = pm.Normal('beta_HasMortgage', mu=0, sigma=10)
    # beta_HasDependents  = pm.Normal('beta_HasDependents', mu=0, sigma=10)
    # beta_LoanPurpose    = pm.Normal('beta_LoanPurpose', mu=0, sigma=10)
    # beta_HasCoSigner    = pm.Normal('beta_HasCoSigner', mu=0, sigma=10)

    # Logistic regression
    linear_combination = (
        beta_0 +
        beta_age             * X_train_subset['Age']+
        beta_income          * X_train_subset['Income']+
        beta_loanAmount      * X_train_subset['LoanAmount']+
        beta_creditScore     * X_train_subset['CreditScore']+
        # beta_monthsEmployed  * X_train_subset['MonthsEmployed'] +
        # beta_numCreditLines  * X_train_subset['NumCreditLines'] +
        beta_interestRate    * X_train_subset['InterestRate']
        # beta_loanTerm        * X_train_subset['LoanTerm'] +
        # beta_DTIRatio        * X_train_subset['DTIRatio'] +
        # beta_Education       * X_train_subset['Education'] +
        # beta_EmploymentType  * X_train_subset['EmploymentType'] +
        # beta_MaritalStatus   * X_train_subset['MaritalStatus'] +
        # beta_HasMortgage     * X_train_subset['HasMortgage'] +
        # beta_HasDependents   * X_train_subset['HasDependents'] +
        # beta_LoanPurpose     * X_train_subset['LoanPurpose'] + 
        # beta_HasCoSigner     * X_train_subset['HasCoSigner']
    )

    # Probability of default
    p = pm.Deterministic('p', pm.math.sigmoid(linear_combination))

    # Likelihood
    observed = pm.Bernoulli('LoanDefault', p=p, observed=y_train_subset)

    # Inference using NUTS (no need for find_MAP or step definition)
    trace = pm.sample(draws=6000, tune=2000, target_accept=0.95, cores=1, chains=4)


# %%

az.plot_trace(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore", "beta_interestRate"])
plt.show()

az.plot_posterior(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore", "beta_interestRate"])
plt.show()


plt.figure(figsize=(12.5, 12.5))

sns.jointplot(
    x=trace.posterior["beta_age"].values.flatten(), 
    y=trace.posterior["beta_loanAmount"].values.flatten(), 
    kind="hex", 
    color="#4CB391"
)

plt.xlabel("beta_age")
plt.ylabel("beta_loanAmount")
plt.show()

# %%

# Extract the posterior samples for p
p_samples = trace.posterior['p'].values  

# Reshape to (samples, observations)
p_samples_reshaped = p_samples.reshape(-1, p_samples.shape[-1])

# Compute the average predicted probability for each observation
y_score = np.mean(p_samples_reshaped, axis=0)

# Plot histogram
plt.figure(figsize=(12.5, 4))
plt.hist(y_score, bins=40, density=True)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of $y_{score}$')
plt.show()

# %%

# Convert probabilities to class predictions (threshold = 0.5)
first_model_prediction = [1 if x > 0.5 else 0 for x in y_score]

# Compute confusion matrix
first_model_confusion_matrix = confusion_matrix(y_train_subset, first_model_prediction)

# Display result
print(first_model_confusion_matrix)

# %%

# Extract posterior samples
beta_0_samples = trace.posterior['beta_0'].values.flatten()
beta_age_samples = trace.posterior['beta_age'].values.flatten()
beta_income_samples = trace.posterior['beta_income'].values.flatten()
beta_loanAmount_samples = trace.posterior['beta_loanAmount'].values.flatten()
beta_creditScore_samples = trace.posterior['beta_creditScore'].values.flatten()
beta_interestRate_samples = trace.posterior['beta_interestRate'].values.flatten()

# Stack all parameter samples
n_samples = len(beta_0_samples)
coefs = np.stack([
    beta_0_samples,
    beta_age_samples,
    beta_income_samples,
    beta_loanAmount_samples,
    beta_creditScore_samples,
    beta_interestRate_samples
], axis=1)  

# Add intercept column to X_test
X_test_subset = X_test_scaled[['Age', 'Income', 'LoanAmount', 'CreditScore', 'InterestRate']].copy()
X_test_augmented = np.hstack([
    np.ones((X_test_subset.shape[0], 1)),  # intercept
    X_test_subset.values
])  

# Compute linear combination
linear_combination_test = coefs @ X_test_augmented.T  

# Apply sigmoid to get probabilities
p_test_samples = 1 / (1 + np.exp(-linear_combination_test))  

# Average over samples to get mean probabilities
y_score_test = np.mean(p_test_samples, axis=0)  

# Convert to binary predictions
first_model_prediction_test = [1 if x > 0.5 else 0 for x in y_score_test]

# Evaluate
print(confusion_matrix(y_test, first_model_prediction_test))







# %% BBayesian Logistic Regression Using MCMC

import pymc as pm

X_train_subset = X_train_scaled[:8000]
y_train_subset = y_train[:8000]


with pm.Model() as bayesian_model:
    # Priors on parameters (use moderately informative priors)
    beta_0              = pm.Normal('beta_0', mu=0, sigma=10)
    beta_age            = pm.Normal('beta_age', mu=0, sigma=10)
    beta_income         = pm.Normal('beta_income', mu=0, sigma=10)
    beta_loanAmount     = pm.Normal('beta_loanAmount', mu=0, sigma=10)
    beta_creditScore    = pm.Normal('beta_creditScore', mu=0, sigma=10)
    # beta_monthsEmployed = pm.Normal('beta_monthsEmployed', mu=0, sigma=10)
    # beta_numCreditLines = pm.Normal('beta_numCreditLines', mu=0, sigma=10)
    # beta_interestRate   = pm.Normal('beta_interestRate', mu=0, sigma=10)
    # beta_loanTerm       = pm.Normal('beta_loanTerm', mu=0, sigma=10)
    # beta_DTIRatio       = pm.Normal('beta_DTIRatio', mu=0, sigma=10)
    # beta_Education      = pm.Normal('beta_Education', mu=0, sigma=10)
    # beta_EmploymentType = pm.Normal('beta_EmploymentType', mu=0, sigma=10)
    # beta_MaritalStatus  = pm.Normal('beta_MaritalStatus', mu=0, sigma=10)
    # beta_HasMortgage    = pm.Normal('beta_HasMortgage', mu=0, sigma=10)
    # beta_HasDependents  = pm.Normal('beta_HasDependents', mu=0, sigma=10)
    # beta_LoanPurpose    = pm.Normal('beta_LoanPurpose', mu=0, sigma=10)
    # beta_HasCoSigner    = pm.Normal('beta_HasCoSigner', mu=0, sigma=10)

    # Logistic regression
    linear_combination = (
        beta_0 +
        beta_age             * X_train_subset['Age']+
        beta_income          * X_train_subset['Income']+
        beta_loanAmount      * X_train_subset['LoanAmount']+
        beta_creditScore     * X_train_subset['CreditScore']
        # beta_monthsEmployed  * X_train_subset['MonthsEmployed'] +
        # beta_numCreditLines  * X_train_subset['NumCreditLines'] +
        # beta_interestRate    * X_train_subset['InterestRate'] +
        # beta_loanTerm        * X_train_subset['LoanTerm'] +
        # beta_DTIRatio        * X_train_subset['DTIRatio'] +
        # beta_Education       * X_train_subset['Education'] +
        # beta_EmploymentType  * X_train_subset['EmploymentType'] +
        # beta_MaritalStatus   * X_train_subset['MaritalStatus'] +
        # beta_HasMortgage     * X_train_subset['HasMortgage'] +
        # beta_HasDependents   * X_train_subset['HasDependents'] +
        # beta_LoanPurpose     * X_train_subset['LoanPurpose'] + 
        # beta_HasCoSigner     * X_train_subset['HasCoSigner']
    )

    # Probability of default
    p = pm.Deterministic('p', pm.math.sigmoid(linear_combination))


with bayesian_model:
    #fit the data 
    observed = pm.Bernoulli('LoanDefault', p=p, observed=y_train_subset)
    start=pm.find_MAP()
    step=pm.Metropolis()

    #samples from posterior distribution 
    trace=pm.sample(25000, step=step, start=start, cores=1)

# %%

# Access the posterior samples
posterior_samples = trace.posterior
burned_trace = posterior_samples.sel(draw=slice(15000, None))

# Plot the trace and posterior distribution
az.plot_trace(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore"])
plt.show()

az.plot_posterior(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore"])
plt.show()

# Example: Check the shape of the burned trace
print(burned_trace)

#%%

az.plot_trace(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore"])
plt.show()

az.plot_posterior(trace, var_names=["beta_0", "beta_age", "beta_income", "beta_loanAmount", "beta_creditScore"])
plt.show()


plt.figure(figsize=(12.5, 12.5))

sns.jointplot(
    x=trace.posterior["beta_age"].values.flatten(), 
    y=trace.posterior["beta_loanAmount"].values.flatten(), 
    kind="hex", 
    color="#4CB391"
)

plt.xlabel("beta_age")
plt.ylabel("beta_loanAmount")
plt.show()

# %%

# Extract the posterior samples for p
p_samples = trace.posterior['p'].values 

# Reshape to (samples, observations)
p_samples_reshaped = p_samples.reshape(-1, p_samples.shape[-1])

# Compute the average predicted probability for each observation
y_score = np.mean(p_samples_reshaped, axis=0)

# Plot histogram
plt.figure(figsize=(12.5, 4))
plt.hist(y_score, bins=40, density=True)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of $y_{score}$')
plt.show()

# %%

# Convert probabilities to class predictions (threshold = 0.5)
first_model_prediction = [1 if x > 0.5 else 0 for x in y_score]

# Compute confusion matrix
first_model_confusion_matrix = confusion_matrix(y_train_subset, first_model_prediction)

# Display result
print(first_model_confusion_matrix)

# %%

# Extract posterior samples
beta_0_samples = trace.posterior['beta_0'].values.flatten()
beta_age_samples = trace.posterior['beta_age'].values.flatten()
beta_income_samples = trace.posterior['beta_income'].values.flatten()
beta_loanAmount_samples = trace.posterior['beta_loanAmount'].values.flatten()
beta_creditScore_samples = trace.posterior['beta_creditScore'].values.flatten()

# Stack all parameter samples
n_samples = len(beta_0_samples)
coefs = np.stack([
    beta_0_samples,
    beta_age_samples,
    beta_income_samples,
    beta_loanAmount_samples,
    beta_creditScore_samples
], axis=1)  

# Add intercept column to X_test
X_test_subset = X_test_scaled[['Age', 'Income', 'LoanAmount', 'CreditScore']].copy()
X_test_augmented = np.hstack([
    np.ones((X_test_subset.shape[0], 1)),  
    X_test_subset.values
])  

# Compute linear combination
linear_combination_test = coefs @ X_test_augmented.T

# Apply sigmoid to get probabilities
p_test_samples = 1 / (1 + np.exp(-linear_combination_test))  

# Average over samples to get mean probabilities
y_score_test = np.mean(p_test_samples, axis=0) 

# Convert to binary predictions
first_model_prediction_test = [1 if x > 0.5 else 0 for x in y_score_test]

# Evaluate
print(confusion_matrix(y_test, first_model_prediction_test))















#%% Neural Network (MLP)

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






















#%% Set Up the Adjacency Matrix & Graph

import networkx as nx

X_train_subset = X_train_scaled[:1000]
y_train_subset = y_train[:1000]

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

# Calculate x% of the possible connections
num_connections = int(0.005 * len(all_pairs))

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


#%% Plot a Subgraph

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


#%% GNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

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
for epoch in range(100):
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

# After training, calculate confusion matrix
model.eval()
pred = out.argmax(dim=1).cpu().numpy()  # Get predicted labels
true_labels = data.y.cpu().numpy()  # True labels

# Generate confusion matrix
cm = confusion_matrix(true_labels, pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#%%

# Step 1: Forward pass to get model predictions
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    out = model(data)  # Perform a forward pass to get the raw output (logits)

# Step 2: Get predicted classes by taking the argmax
predicted_classes = out.argmax(dim=1)

# Step 3: Find connected nodes (already done)
connected_columns = adj_matrix.columns[(adj_matrix > 0).any(axis=0)].tolist()
connected_node_indices = [int(node) for node in connected_columns]

# Step 4: Map connected node indices to the corresponding row indices in the original dataset
node_index_mapping = {node: idx for idx, node in enumerate(G.nodes())}

# Create a list of row indices for the connected nodes
connected_row_indices = [node_index_mapping[node] for node in connected_node_indices]

# Step 5: Get the predicted classes for these connected nodes
predicted_classes_connected = predicted_classes[connected_row_indices]

# Step 6: Get the true labels for these connected nodes
true_labels_connected = y_train_subset.iloc[connected_row_indices].values

# Step 7: Calculate accuracy for the connected nodes subgroup
correct_predictions = (predicted_classes_connected.numpy() == true_labels_connected)
accuracy_connected = correct_predictions.mean()

print(f"Accuracy on the connected loan default nodes: {accuracy_connected}")

# Step 8: Generate the confusion matrix for the connected nodes
cm = confusion_matrix(true_labels_connected, predicted_classes_connected.numpy())

# Step 9: Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix (Connected Nodes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#%%


































#%% Transformers Preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your data
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

df = df.drop(columns=['LoanID'])

# Check results
print(df['Default'].value_counts())
print(f"Total samples: {len(df)}")

# Encode categorical columns (categorical columns have dtype 'object')
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and labels
X_train = train_data.drop('Default', axis=1)
y_train = train_data['Default']
X_test = test_data.drop('Default', axis=1)
y_test = test_data['Default']

# Get unique values for each categorical feature
cat_cardinalities = [df[col].nunique() for col in categorical_columns]
cat_features_info = [(cardinality, 32) for cardinality in cat_cardinalities]  # embedding dim fixed to 32

# Preprocessing
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in categorical_columns]

# Encode categorical features as integers
for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Resplit after encoding
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_data.drop('Default', axis=1)
y_train = train_data['Default']
X_test = test_data.drop('Default', axis=1)
y_test = test_data['Default']

# Standardize numerical features
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])


# %% Define TransformerBlock

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

# %% Define TabTransformer

class TabTransformer(keras.Model):
    def __init__(self, num_features, cat_features_info, num_classes, embed_dim=32, num_heads=2, num_transformer_blocks=2, mlp_units=[64, 32]):
        super(TabTransformer, self).__init__()
        self.embeddings = [layers.Embedding(input_dim=cat[0], output_dim=embed_dim) for cat in cat_features_info]
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_transformer_blocks)]
        self.flatten = layers.Flatten()
        self.dense_layers = [layers.Dense(units, activation='relu') for units in mlp_units]
        self.output_layer = layers.Dense(1, activation='sigmoid')
        self.num_features = num_features

    def call(self, inputs):
        cat_inputs, num_inputs = inputs
        x_cat = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = tf.stack(x_cat, axis=1)  # Shape: (batch_size, num_cat_features, embed_dim)

        for transformer_block in self.transformer_blocks:
            x_cat = transformer_block(x_cat)

        x_cat_flat = self.flatten(x_cat)
        x = tf.concat([x_cat_flat, num_inputs], axis=1)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.output_layer(x)


# %% Feed Data

# Convert to tensors
cat_data_train = tf.convert_to_tensor(X_train[categorical_columns].values, dtype=tf.int32)
num_data_train = tf.convert_to_tensor(X_train[numerical_columns].values, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)

cat_data_test = tf.convert_to_tensor(X_test[categorical_columns].values, dtype=tf.int32)
num_data_test = tf.convert_to_tensor(X_test[numerical_columns].values, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# %% Train the TabTransformer

model = TabTransformer(num_features=len(numerical_columns),
                       cat_features_info=cat_features_info,
                       num_classes=1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit((cat_data_train, num_data_train), y_train_tensor, epochs=10, batch_size=32, validation_split=0.2)

# %% Prediction

# Get predicted probabilities and convert to class labels (0 or 1)
y_pred_probs = model.predict((cat_data_test, num_data_test))
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Compute confusion matrix
cm = confusion_matrix(y_test_tensor.numpy(), y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test_tensor.numpy(), y_pred, digits=4))
