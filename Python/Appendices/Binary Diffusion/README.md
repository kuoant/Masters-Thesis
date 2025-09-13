# How to run it:

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

