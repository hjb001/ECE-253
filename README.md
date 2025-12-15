# FFT-Based Image Denoising

This repo contains several lightweight experiments for denoising smartphone photos from the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) dataset. It includes a classical Gaussian baseline and two FFT-based neural networks that predict noise directly in the frequency domain.

## Repo Layout

- `data/dldata.py` – downloads the Kaggle *smartphone-image-denoising-dataset* into the project.
- `guissian_denoising.py` – Gaussian-blur baseline plus PSNR evaluation/saving utilities.
- `learnable_denoising.py` – first FFT ConvNet denoiser.
- `learnable_denoising_v2.py` – deeper residual FFT denoiser with AMP/MPS support and richer training loop.
- `fft_results*`, `gaussian_denoise_results`, `real_test_denoise_results` – example output folders created by the scripts.

## Prerequisites

- Python 3.9+ (works on macOS, Linux, and Windows).
- GPU with CUDA is optional; scripts automatically fall back to MPS or CPU.
- Python packages:

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm kagglehub h5py
```

## Getting the Dataset

1. Create a Kaggle API token (see [Kaggle docs](https://www.kaggle.com/docs/api)).
2. Place `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
3. Run the helper to download/unpack SIDD Small into the repo:

```bash
python data/dldata.py
```

This script stores the Kaggle cache inside `./kagglehub_cache` and copies the dataset to `./data/SIDD_Small_sRGB_Only/Data`. If you prefer another directory, change `target_dir` or point the `data_root` variables inside the training scripts to your dataset path.

## Running the Baseline (Gaussian Blur)

```bash
python guissian_denoising.py
```

- Loads a small subset of SIDD, applies `cv2.GaussianBlur`, prints per-image PSNR, and writes denoised PNGs to `gaussian_denoise_results/`.
- Adjust kernel size and sigma via the script arguments (`gaussian_denoise(noisy, ksize=7, sigma=2)`).

## Training the FFT Denoisers

Both training scripts expect the same dataset layout.

### Minimal FFT Network

```bash
python learnable_denoising.py
```

- Uses a 3-layer FFT convolutional model.
- Saves curves to `fft_results/loss.png` and `fft_results/psnr.png`.
- Modify `epochs`, `batch_size`, and `data_root` inside the file as needed.

### Residual FFT Network (recommended)

```bash
python learnable_denoising_v2.py
```

- Adds residual FFT blocks, mixed precision on CUDA, Apple Silicon-aware DataLoader settings, and best-model checkpointing (`fft_results_v2/best_model.pth`).
- Configure via the `main()` function: `epochs`, `lr`, `output_dir`, etc.

## Tips

- Use `wandb/` for experiment tracking if desired (hook up manually inside the scripts).
- For custom data, replace `data_root` and ensure images are normalized to `[0, 1]`.
- When running on CPU/MPS, reduce `batch_size` if you experience memory pressure.

## License

No explicit license file is provided; treat this as personal research code unless the owner specifies otherwise.
