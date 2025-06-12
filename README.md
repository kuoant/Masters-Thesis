# Master's Thesis

This repository contains the code, data, and literature references for my Master's Thesis at the University of Basel.

## Repository Structure

- **Literature/**  
  Contains research papers and references grouped by topic:
  - `Diffusion Models/`
  - `Embeddings/`
  - `GNN/`
  - `Transformers/`
  - `XGBoost/`
- **Python/**  
  Main source code and data for experiments and analysis:
  - `environment.yaml` & `requirements.txt`: Python environment and dependencies.
  - `Loan_default.csv`: Dataset used for experiments.
  - `Loan_default.py`: Main script for data analysis or modeling.
  - `.pt_tmp/`: Temporary files or model checkpoints.
  - `Binary Diffusion/`, `Demo/`, `ETL Process/`, `Housing/`: Submodules or experiments related to the Appendix.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/masters-thesis.git
   cd masters-thesis/Python
   ```

2. **Set up the environment**
   ```sh
   conda env create -f environment.yaml
   conda activate <your-environment-name>
   ```

3. **Run the main script**
   ```sh
   python Loan_default.py
   ```

## Project Overview

This project explores advanced machine learning models for tabular data, including diffusion models, embeddings, GNNs, transformers, and XGBoost.  
Relevant literature is organized in the `Literature/` folder.

## License

This repository is for academic use only.

## Contact

For questions, please contact [fabian . kuonen _at_ unibas.ch].