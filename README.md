# Master's Thesis – University of Basel

This repository contains all code, data, and literature references for my Master's Thesis at the University of Basel.

## Repository Structure

```
Masters Thesis/
├── Literature/
│   ├── Literature Diffusion Models/
│   ├── Literature Embeddings/
│   ├── Literature GNN/
│   ├── Literature Transformers/
│   └── Literature XGBoost/
├── Python/
│   ├── .pt_tmp/
│   ├── Appendices/
│   │   ├── Binary Diffusion/
│   │   ├── California Housing/
│   │   ├── Demo/
│   │   └── ETL Process/
│   ├── Modular Version/
│   │   ├── Modular Solution GNN/
│   │   │   ├── data/
│   │   │   ├── outputs/
│   │   │   └── src/
│   │   └── Modular Solution Transformer/
│   │       ├── data/
│   │       ├── outputs/
│   │       └── src/
│   ├── Scripted Version/
│   │   ├── GNN/
│   │   │   └── data/
│   │   ├── main.py
│   │   └── Transformer/
│   ├── environment.yaml
│   └── requirements.txt
├── .gitignore
├── README.md
└── Loan_default.py
```

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/kuoant/masters-thesis.git
   cd "Masters Thesis/Python"
   ```

2. **Set up the environment**
   ```sh
   conda env create -f environment.yaml
   conda activate <your-environment-name>
   ```

3. **Install additional requirements (if needed)**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run main scripts**
   - For the loan default experiment:
     ```sh
     python Loan_default.py
     ```
   - For GNN/Transformer experiments, see the respective folders under `Modular Version` and `Scripted Version`.

## Project Overview

This project investigates advanced machine learning models for tabular data, including:
- Diffusion Models
- Embeddings
- Graph Neural Networks (GNN)
- Transformers
- XGBoost

Relevant literature is organized in the `Literature/` folder.  
Code and experiments are organized in the `Python/` folder, with modular and scripted versions for different model architectures.

## License

This repository is for academic and research purposes only.

## Contact

For questions, please contact [fakuonen _at_ ethz.ch].