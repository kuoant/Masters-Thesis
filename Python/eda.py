#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Scripted Version/GNN/data/Loan_default.csv")

# General summary of the dataset
print("=== DataFrame Info ===")
df.info()

print("\n=== Summary Statistics (Numerical Columns) ===")
print(df.describe())

print("\n=== Summary Statistics (Categorical Columns) ===")
print(df.describe(include=['object', 'category']))

print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

print("\n=== Target Variable Distribution ===")
print(df['Default'].value_counts(normalize=True))



# %%
sns.set_theme(style="whitegrid")

# Plot 1: Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Default", palette="pastel")
plt.title("Loan Default Distribution")
plt.xticks([0, 1], ["No Default", "Default"])
plt.ylabel("Count")
plt.xlabel("Default")
plt.tight_layout()
plt.show()

# Plot 2: Distribution of credit scores by default status
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="CreditScore", hue="Default", kde=True, element="step", stat="density", common_norm=False)
plt.title("Credit Score Distribution by Default Status")
plt.xlabel("Credit Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# Plot 3: Violin plot of Debt-to-Income Ratio by Default
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="Default", y="DTIRatio", palette="viridis")
plt.title("Debt-to-Income Ratio by Default")
plt.xticks([0, 1], ["No Default", "Default"])
plt.tight_layout()
plt.show()

# Plot 4: Default rate by Loan Purpose
plt.figure(figsize=(10, 6))
purpose_default_rate = df.groupby("LoanPurpose")["Default"].mean().sort_values(ascending=False)
sns.barplot(x=purpose_default_rate.values, y=purpose_default_rate.index, palette="muted")
plt.title("Default Rate by Loan Purpose")
plt.xlabel("Default Rate")
plt.ylabel("Loan Purpose")
plt.tight_layout()
plt.show()
# %%
