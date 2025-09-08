#====================================================================================================================
# Imports
#====================================================================================================================
#%%

# Core Libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing & Statistics
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

#====================================================================================================================
# Colors
#====================================================================================================================

# Set global visualization style and color palette
cubehelix_palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.8)
sns.set_palette(cubehelix_palette)
sns.set_style("whitegrid")

# Apply cubehelix palette to matplotlib color cycle
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cubehelix_palette)

# Define cubehelix colormap for heatmaps
cubehelix_cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

#====================================================================================================================
# Data Preprocessing
#====================================================================================================================

# Load dataset
df = pd.read_csv("Scripted Version/GNN/data/Loan_default.csv")
print(df.head())

# Remove LoanID as it's just an identifier
df = df.drop('LoanID', axis=1)

#====================================================================================================================
# Exploratory Data Analysis (EDA)
#====================================================================================================================

## 1. Data Quality Analysis
print("\n Data Quality Checks")
# Check for duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Constant columns: {constant_cols}")

## 2. Statistical Normality Tests
print("\n Normality Tests for Numerical Features")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
analysis_num_cols = [col for col in numeric_cols if col != 'Default']

alpha = 0.05
bonferroni_alpha = alpha / len(analysis_num_cols)

print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

# Shapiro-Wilk test
for col in analysis_num_cols:
    stat, p = stats.shapiro(df[col].sample(min(5000, len(df)), random_state=42))
    interpretation = "(Normal)" if p > bonferroni_alpha else "(Non-normal)"
    print(f"{col}: Statistics={stat:.3f}, p-value={p:.3f} {interpretation}")

## 3. Correlation Analysis
print("\n Correlation Analysis")
# Calculate correlation matrix
corr_matrix = df[analysis_num_cols + ['Default']].corr(method='spearman')

cubehelix_cmap = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.8, as_cmap=True)
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".6f",
    cmap=cubehelix_cmap,
    center=0,
    vmin=-0.01,
    vmax=0.01
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identify high correlations
high_corr = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns 
             if i != j and abs(corr_matrix.loc[i, j]) > 0.7]
print(f"Highly correlated features (|r| > 0.7): {high_corr}")

## 4. Statistical Significance Testing
print("\n Group Differences (Default vs Non-Default)")
alpha = 0.05
bonferroni_alpha = alpha / len(analysis_num_cols)
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

for col in analysis_num_cols:
    group1 = df[df['Default'] == 0][col].dropna()
    group2 = df[df['Default'] == 1][col].dropna()
    
    # Mann-Whitney U test (non-parametric)
    stat, p = stats.mannwhitneyu(group1, group2)
    
    interpretation = "(Significant)" if p < bonferroni_alpha else "(Not significant)"
    print(f"{col}: U-stat={stat:.1f}, p-value={p:.4f} {interpretation}")

## 5. Effect Size Analysis
print("\n Effect Size Analysis")
for col in analysis_num_cols:
    group1 = df[df['Default'] == 0][col].dropna()
    group2 = df[df['Default'] == 1][col].dropna()
    
    # Cohen's d
    pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
    d = (group1.mean() - group2.mean()) / pooled_std
    print(f"{col}: Cohen's d={d:.3f} {'(Small)' if abs(d) < 0.5 else '(Medium)' if abs(d) < 0.8 else '(Large)'}")

## 6. Categorical Analysis with Chi-square
print("\n Categorical Feature Analysis")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
alpha = 0.05
bonferroni_alpha = alpha / len(categorical_cols)
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df['Default'])
    if (contingency_table.values < 5).sum() > 0:
        print(f"{col}: Cannot perform chi-square - some categories have low counts")
        continue
        
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    interpretation = "(Significant)" if p < bonferroni_alpha else "(Not significant)"
    print(f"{col}: Chi2={chi2:.1f}, p-value={p:.4f} {interpretation}")

## 7. Outlier Detection
print("\n Outlier Detection")
for col in analysis_num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

## 8. Feature Importance Analysis
print("\n Feature Importance Analysis")
# Encode categorical variables
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

# Calculate mutual information
X = df_encoded.drop('Default', axis=1)
y = df_encoded['Default']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Plot feature importance
n = len(mi_scores)
continuous_palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.8, n_colors=n)
continuous_palette = continuous_palette[::-1]

plt.figure(figsize=(10, 6))
mi_scores.plot(kind='barh', color=continuous_palette)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

## 9. Pairplot with Hue
sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df.copy()
pairplot_palette = sns.cubehelix_palette(
    start=0.5, rot=-0.5,
    dark=0.6, light=0.3,
    n_colors=2,
    reverse=True
)
g = sns.pairplot(
    sample_df,
    vars=analysis_num_cols[:5],
    hue='Default',
    palette=pairplot_palette,
    corner=True,
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'}
)

if g._legend is not None:
    g._legend.set_title("Default", prop={'size': 20})
    for text in g._legend.get_texts():
        text.set_fontsize(18)
    g._legend.set_bbox_to_anchor((1., 1.))

for ax in g.axes.flatten():
    if ax is not None: 
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

## 10. Default rate by categorical features
combo_df = (
    df.groupby(['HasCoSigner', 'Education'])['Default']
    .agg(['count', 'sum'])
    .rename(columns={'count': 'Total', 'sum': 'Defaults'})
    .reset_index()
)
combo_df['DefaultRate'] = combo_df['Defaults'] / combo_df['Total']

# Display as table
print("\n Default rates by HasCoSigner and Education:")
print(combo_df[['HasCoSigner', 'Education', 'DefaultRate']].sort_values(by='DefaultRate', ascending=False))

# Visualize with heatmap
pivot_df = combo_df.pivot(index='HasCoSigner', columns='Education', values='DefaultRate')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, fmt=".2%", cmap=cubehelix_cmap, linewidths=0.5)
plt.tight_layout()
plt.show()

# Default rate by each categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    default_rates = df.groupby(col)['Default'].mean().sort_values()
    n_cats = len(default_rates)
    darker_palette = sns.cubehelix_palette(
        start=.5, rot=-.5, 
        dark=0.7, light=0.3, 
        n_colors=n_cats, 
        reverse=True
    )
    sns.barplot(
        x=col, 
        y='Default', 
        data=df, 
        ci=None, 
        palette=darker_palette,
        order=default_rates.index 
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# %%
