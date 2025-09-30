# Representation Learning for Downstream Statistical Modeling  
### A Simulation Framework for Studying GNN and Transformer Embeddings in Gradient Boosted Tabular Classification  

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Frameworks](https://img.shields.io/badge/frameworks-PyTorch%20Geometric%2C%20Transformers%2C%20XGBoost-orange)]()

Master's Thesis — Data Science & Computational Economics  
Chair of Econometrics & Statistics, University of Basel  
Supervised by: Prof. Christian Kleiber

---

## 📖 Abstract
This project investigates whether **boosting methods** such as XGBoost can be enhanced by integrating **graph neural networks (GNNs)** and **transformer-based embeddings** when simulated graph or text structures complement tabular data. The hybrid framework first learns low-dimensional representations from GNNs or transformers and subsequently feeds them into an XGBoost classifier.  

The empirical results show that this approach outperforms boosting models trained solely on raw features, with average performance improvements from ~0.70 to >0.85 across multiple metrics (Accuracy, Precision, Recall, F1, AUC). Robustness holds across varying levels of graph connectivity and textual noise.  
Two applied case studies in econometrics — California Housing (GNN embeddings) and Twitter sentiment (transformer embeddings) — demonstrate the versatility of the approach in domains traditionally dominated by statistical methods.  

---

## 📂 Project Structure
```
Python/
│
├── Appendices/
│   ├── Binary Diffusion/         # Auxiliary simulation project
│   └── ETL Process/              # Standalone ETL pipeline
│
├── Applications/
│   ├── California Housing/       # GNN application on econometrics data
│   └── Twitter/                  # Transformer application on sentiment data
│
├── Scripted Version/
│   ├── GNN/                      # Graph simulation framework
│   └── Transformer/              # Text simulation framework
│
├── EDA.py                        # Exploratory data analysis
├── Proof_of_Concept.py           # Initial PoC
├── requirements.txt              # Required packages
├── environment.yml               # Main Conda environment
└── README.md                     # This file
```

---

## ⚙️ Setup

### Prerequisites
- Python ≥ 3.8  
- [Anaconda](https://www.anaconda.com/) (recommended)  

### Installation
Clone the repository:
```bash
git clone https://github.com/kuoant/Masters-Thesis.git
cd Masters-Thesis
```

Create the environment:
```bash
conda env create -f environment.yml
conda activate master_thesis
```

Additional environment.yml files are provided in application subfolders for California Housing and Twitter sentiment analysis.

---

## 🚀 Usage
Each module can be run independently:

```bash
# Run graph simulation experiments
cd Scripted\ Version/GNN
python main.py

# Run text simulation experiments
cd Scripted\ Version/Transformer
python main.py

# Run California Housing application
cd Applications/California\ Housing
python housing.py

# Run Twitter sentiment application
cd Applications/Twitter
python twitter.py
```

The ETL project and Binary Diffusion are standalone as well and not required for reproducing thesis experiments.

---

## 📊 Results (Summary)
Hybrid models (XGBoost + embeddings) outperform baseline XGBoost:

- Loan default detection: performance improved from ~0.70 → >0.85 across all metrics
- Gains remain stable under varying graph connectivity and textual noise
- California Housing: GNN-based embeddings improved predictive accuracy in econometric regression-to-classification setting
- Twitter sentiment: Transformer embeddings improved sentiment classification beyond traditional text preprocessing

---

## 📚 Methodology
**Representation Learning**
- GNNs (PyTorch Geometric) for graph-structured data
- Transformers (HuggingFace) for textual data

**Downstream Model**
- XGBoost classifier trained on concatenated raw + embedding features

**Simulation Framework**
- Generates controlled tabular, graph, and text structures
- Allows studying robustness under varying connectivity and noise

---

## 📦 Datasets
- Loan Default (Kaggle, by Nikhil) → extended with simulated graph & text structures
- California Housing → GNN-based econometric application  
- Twitter Sentiment → Transformer-based econometric application

Sample data included in repository (`/data/` folders).

---

## 📖 Citation
If you use this code or framework, please cite as:

```bibtex
@mastersthesis{kuonen2025representation,
  title        = {Representation Learning for Downstream Statistical Modeling: 
                  A Simulation Framework for Studying GNN and Transformer Embeddings 
                  in Gradient Boosted Tabular Classification},
  author       = {Fabian Gabriel Kuonen},
  school       = {University of Basel},
  year         = {2025},
  type         = {Master's Thesis}
}
```

**Related works:**
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
- Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS.

---

## 🏛️ Acknowledgements
This work was conducted as part of the Master's Program in Data Science & Computational Economics at the University of Basel, Chair of Econometrics & Statistics, supervised by Prof. Christian Kleiber.

---

## 📬 Contact
**Fabian Kuonen**  
📧 fakuonen@ethz.ch