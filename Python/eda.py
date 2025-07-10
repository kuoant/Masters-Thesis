#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Scripted Version/GNN/data/Loan_default.csv")

# Remove LoanID as it's just an identifier
df = df.drop('LoanID', axis=1)

## 1. Advanced Data Quality Analysis
print("\n=== Advanced Data Quality Checks ===")
# Check for duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Constant columns: {constant_cols}")

## 2. Statistical Normality Tests
print("\n=== Normality Tests for Numerical Features ===")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Remove Default from numeric cols for analysis
analysis_num_cols = [col for col in numeric_cols if col != 'Default']

for col in analysis_num_cols:
    stat, p = stats.shapiro(df[col].sample(min(5000, len(df))))  # Shapiro-Wilk test with sampling
    print(f"{col}: Statistics={stat:.3f}, p-value={p:.3f} {'(Normal)' if p > 0.05 else '(Non-normal)'}")

## 3. Correlation Analysis
print("\n=== Correlation Analysis ===")
# Calculate correlation matrix
corr_matrix = df[analysis_num_cols + ['Default']].corr(method='spearman')

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix (Spearman)")
plt.tight_layout()
plt.show()

# Identify high correlations
high_corr = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns 
             if i != j and abs(corr_matrix.loc[i, j]) > 0.7]
print(f"Highly correlated features (|r| > 0.7): {high_corr}")

## 4. Statistical Significance Testing
print("\n=== Group Differences (Default vs Non-Default) ===")
for col in analysis_num_cols:
    group1 = df[df['Default'] == 0][col].dropna()
    group2 = df[df['Default'] == 1][col].dropna()
    
    # Mann-Whitney U test (non-parametric)
    stat, p = stats.mannwhitneyu(group1, group2)
    print(f"{col}: U-stat={stat:.1f}, p-value={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

## 5. Effect Size Analysis
print("\n=== Effect Size Analysis ===")
for col in analysis_num_cols:
    group1 = df[df['Default'] == 0][col].dropna()
    group2 = df[df['Default'] == 1][col].dropna()
    
    # Cohen's d
    pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
    d = (group1.mean() - group2.mean()) / pooled_std
    print(f"{col}: Cohen's d={d:.3f} {'(Small)' if abs(d) < 0.5 else '(Medium)' if abs(d) < 0.8 else '(Large)'}")

## 6. Categorical Analysis with Chi-square
print("\n=== Categorical Feature Analysis ===")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Default'])
    # Skip if any expected counts are too low for chi-square
    if (contingency_table.values < 5).sum() > 0:
        print(f"{col}: Cannot perform chi-square - some categories have low counts")
        continue
        
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"{col}: Chi2={chi2:.1f}, p-value={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")

## 7. Outlier Detection
print("\n=== Outlier Detection ===")
for col in analysis_num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

## 8. Feature Importance Analysis (Preliminary)
print("\n=== Feature Importance Analysis ===")
# Encode categorical variables
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

# Calculate mutual information
from sklearn.feature_selection import mutual_info_classif
X = df_encoded.drop('Default', axis=1)
y = df_encoded['Default']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
mi_scores.plot(kind='barh', color='teal')
plt.title("Mutual Information Scores")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

## 9. Advanced Visualization: Pairplot with Hue
print("\n=== Pairwise Relationships ===")
# Sample the data for visualization if large
sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df.copy()
sns.pairplot(sample_df, vars=analysis_num_cols[:5], hue='Default', palette='viridis', corner=True)
plt.suptitle("Pairwise Relationships by Default Status", y=1.02)
plt.show()

## 10. Additional Important Visualizations
# Default rate by categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.barplot(x=col, y='Default', data=df, ci=None)
    plt.title(f"Default Rate by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# %%
