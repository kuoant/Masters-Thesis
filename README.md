# Master's Thesis Project: Representation Learning for Downstream Statistical Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)

This repository contains the complete implementation and analysis for my Master's Thesis at the University of Basel, focusing on advanced machine learning approaches (GNN, transformers & XGBoost) for tabular data analysis, with an emphasis on loan default prediction.

## 🎯 Project Overview

This research investigates the application of state-of-the-art machine learning models to tabular data, comparing state-of-the-art enriched with representations from modern deep learning architectures:

- **Graph Neural Networks (GNNs)** for capturing graph relationships
- **Transformer architectures** adapted for text in tabular data
- **Diffusion Models** for synthetic data generation
- **XGBoost** as baseline comparison

### Key Research Questions
- How do modern deep learning approaches compare to traditional methods on tabular data?
- Can Graph Neural Networks effectively model feature relationships in loan default prediction?
- What is the impact of different preprocessing and embedding strategies?

## 📁 Repository Structure

```

```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda or Miniconda
- CUDA-compatible GPU (recommended for deep learning models), but works also with CPU (by default)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kuoant/masters-thesis.git
   cd "Masters Thesis"
   ```

2. **Set up the environment**
   ```bash
   cd Python
   conda env create -f environment.yaml
   conda activate thesis-env
   ```

3. **Install additional dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

#### Exploratory Data Analysis
```bash
python eda.py
```

#### Main Loan Default Analysis
```bash
python Loan_default.py
```

#### GNN Experiments
```bash
cd "Scripted Version/GNN"
python main.py
```

#### Modular Implementations
```bash
# For GNN
cd "Modular Version/Modular Solution GNN"
python -m src.main

# For Transformer
cd "Modular Version/Modular Solution Transformer"
python -m src.main
```

## 📊 Dataset

The primary dataset used is a loan default prediction dataset containing:
- **Features**: Borrower demographics, loan characteristics, financial metrics
- **Target**: Binary default indicator
- **Size**: [Specify number of samples and features]
- **Source**: [Specify data source if public]

Additional datasets used for comparative analysis:
- California Housing dataset (regression benchmark)
- Synthetic datasets generated through diffusion models

## 🔬 Methodology

### Data Preprocessing
- Comprehensive exploratory data analysis (EDA)
- Statistical normality testing
- Outlier detection and treatment
- Feature scaling and normalization
- Categorical encoding strategies

### Model Architectures

#### Graph Neural Networks
- Node feature embedding
- Graph construction from tabular features
- Message passing mechanisms
- Graph-level predictions

#### Transformers for Tabular Data
- Feature tokenization
- Positional encoding adaptations
- Attention mechanism analysis
- Comparative attention visualization

#### Baseline Models
- XGBoost with hyperparameter tuning
- Random Forest ensembles
- Logistic regression with regularization

### Evaluation Metrics
- Classification accuracy
- Precision, Recall, F1-score
- AUC-ROC and AUC-PR
- Feature importance analysis
- Model interpretability assessment

## 📈 Key Results

*[This section would be populated with your actual results]*

- GNN performance: [Accuracy/F1 scores]
- Transformer performance: [Accuracy/F1 scores]
- XGBoost baseline: [Accuracy/F1 scores]
- Feature importance rankings
- Computational efficiency comparison

## 🛠️ Code Organization

### Modular Version
Production-ready, well-structured code with:
- Separate data processing modules
- Model training and evaluation pipelines
- Configuration management
- Comprehensive logging

### Scripted Version
Research and experimentation scripts:
- Rapid prototyping
- Hyperparameter exploration
- Ablation studies
- Visualization generation

## 📚 Literature Review

The `Literature/` folder contains categorized research papers covering:
- **Diffusion Models**: Generative modeling for tabular data
- **Embeddings**: Advanced categorical and numerical encoding
- **GNNs**: Graph-based approaches for structured data
- **Transformers**: Attention mechanisms in non-sequential data
- **XGBoost**: Gradient boosting optimization techniques

## 🔧 Technical Requirements

**Hardware:**
- RAM: 16GB+ recommended
- GPU: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- Storage: 10GB+ free space

**Software:**
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Scikit-learn
- XGBoost
- Pandas, NumPy, Matplotlib, Seaborn

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{kuonen2025tabular,
  title={Advanced Machine Learning Approaches for Tabular Data Analysis},
  author={Fabian Kuonen},
  school={University of Basel},
  year={2025},
  type={Master's Thesis}
}
```

## 🤝 Contributing

This is an academic research project. For questions, suggestions, or collaboration opportunities, please reach out via email.

## 📄 License

This repository is for academic and research purposes only. The code and methodology are available for educational use and replication studies.

## 📧 Contact

**Fabian Kuonen**  
Master's Student, University of Basel  
For questions, please contact [fakuonen _at_ ethz.ch].

---

*This repository represents ongoing research in machine learning for tabular data. Results and methodologies are subject to review and validation.*

