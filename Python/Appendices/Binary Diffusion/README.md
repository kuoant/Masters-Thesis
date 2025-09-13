## How to run it:

1. Activate the environment:

    ```bash
    conda activate binary_diffusion
    ```

2. Navigate to the binary diffusion folder:

    ```bash
    cd "/your_path_to/binary-diffusion-tabular"
    ```

3. Generate new samples:

    ```bash
    python sample.py \
      --ckpt="/your_path_to/Masters Thesis/Python/Appendices/Binary Diffusion/results/adult_CFG_small/model-final.pt" \
      --ckpt_transformation="/your_path_to/Masters Thesis/Python/Appendices/Binary Diffusion/results/adult_CFG_small/transformation.joblib" \
      --n_timesteps=100 \
      --out="/your_path_to/Masters Thesis/Python/Appendices/Binary Diffusion/samples" \
      --n_samples=1000 \
      --batch_size=64 \
      --threshold=0.5 \
      --strategy=target \
      --seed=42 \
      --guidance_scale=2.0 \
      --target_column_name=income \
      --device=cpu
    ```

## Credits

Sample generation builds on the work of **Vitaliy Kinakh** and **Slava Voloshynovskiy**:

- **Paper:** *Tabular Data Generation using Binary Diffusion*  
  - **arXiv:** [arXiv:2409.13882v2](https://arxiv.org/abs/2409.13882)  
  - **DOI:** [10.48550/arXiv.2409.13882](https://doi.org/10.48550/arXiv.2409.13882)  
  - **Summary:** Generating synthetic tabular data is critical in machine learning, especially when real data is limited or sensitive. Traditional generative models often face challenges due to mixed data types and varied distributions, requiring complex preprocessing or large pretrained models. This paper introduces a **lossless binary transformation method** converting any tabular data into fixed-size binary representations, along with a new generative model called **Binary Diffusion**, specifically designed for binary data. Binary Diffusion leverages XOR operations for noise addition/removal and uses binary cross-entropy loss for training, eliminating the need for extensive preprocessing or large pretrained models. The method outperforms existing state-of-the-art models on benchmark datasets such as Travel, Adult Income, and Diabetes while being significantly smaller in size.  
  - **Accepted at:** 3rd Table Representation Learning Workshop @ NeurIPS 2024

- **Code and Models:** Available on GitHub: [https://github.com/vkinakh/binary-diffusion-tabular](https://github.com/vkinakh/binary-diffusion-tabular)

