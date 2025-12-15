import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from Denoising.data.learnable_denoising_v2 import FFTDenoiser, calculate_psnr, get_device


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the trained FFTDenoiser and run inference on images inside a folder. "
            "The script optionally computes PSNR when ground-truth files are available."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/Users/huangjunbo/Downloads/ECE-253/fft_results_v2/best_model.pth",
        help="Path to the checkpoint produced during training.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/real_Test_256",
        help="Folder that stores noisy input images.",
    )
    parser.add_argument(
        "--output-dir",
        default="real_test_denoise_results",
        help="Folder used to save denoised images and metrics.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help=(
            "Optional folder that stores ground-truth images. "
            "If omitted, the script tries to locate *_GT* files next to each noisy image."
        ),
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help=(
            "Resize all inputs to this size before denoising. "
            "Use --no-resize to keep original resolution."
        ),
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Keep images at their original resolution regardless of --resize.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force running on 'cpu', 'cuda', or 'mps'. Default auto-detects (CUDA > MPS > CPU).",
    )
    parser.add_argument(
        "--gaussian-ksize",
        type=int,
        default=5,
        help="Odd kernel size (pixels) for baseline Gaussian denoising.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Standard deviation for baseline Gaussian denoising.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib plots (enabled by default).",
    )
    return parser.parse_args()


def load_image_tensor(
    image_path: Path, resize_hw: Optional[Tuple[int, int]]
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load image as tensor in [0,1], optionally resizing; returns tensor and original size (H, W)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    if resize_hw is not None:
        target_h, target_w = resize_hw
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Resize dimensions must be positive integers.")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w)


def tensor_to_image(tensor: torch.Tensor, dst_size: Optional[Tuple[int, int]]) -> np.ndarray:
    """Convert BCHW tensor in [0,1] to uint8 BGR image, resizing if requested."""
    img = tensor.squeeze(0).clamp_(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if dst_size is not None:
        h, w = dst_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img


def collect_images(folder: Path) -> List[Path]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if not files:
        raise FileNotFoundError(f"No supported image files found inside {folder}")
    return files


def gaussian_denoise_tensor(noisy_tensor: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    """
    Apply a Gaussian blur baseline to the input tensor (BCHW, values in [0,1]).
    Returns a tensor on the same device.
    """
    blurred = noisy_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    blurred = cv2.GaussianBlur(blurred, (ksize, ksize), sigmaX=sigma)
    blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0)
    return blurred_tensor.to(noisy_tensor.device)


def estimate_snr(noisy_tensor: torch.Tensor, denoised_tensor: torch.Tensor) -> float:
    """
    Estimate SNR without GT by treating (noisy - denoised) as residual noise.
    Returns SNR in dB; very large value if residual noise is near zero.
    """
    diff = noisy_tensor - denoised_tensor
    noise_power = diff.pow(2).mean().item()
    signal_power = denoised_tensor.pow(2).mean().item()
    if noise_power <= 1e-12:
        return float("inf")
    if signal_power <= 1e-12:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def mean_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """Average absolute difference between two BCHW tensors."""
    return torch.mean(torch.abs(a - b)).item()


def plot_metric_curves(metrics: List[dict], output_dir: Path) -> None:
    """Plot PSNR/SNR/MAE curves comparing noisy, Gaussian, and FFT denoisers."""
    if not metrics:
        return

    image_names = [item["filename"] for item in metrics]
    indices = np.arange(len(image_names), dtype=float)

    def to_array(key: str, allow_none: bool = True) -> np.ndarray:
        values = []
        for item in metrics:
            val = item.get(key)
            if val is None and allow_none:
                values.append(np.nan)
            elif isinstance(val, float) and math.isinf(val):
                values.append(np.nan)
            else:
                values.append(val if val is not None else np.nan)
        return np.array(values, dtype=float)

    noisy_psnr = to_array("noisy_psnr")
    fft_psnr = to_array("fft_psnr")
    gaussian_psnr = to_array("gaussian_psnr")
    fft_snr = to_array("fft_snr_est", allow_none=False)
    gaussian_snr = to_array("gaussian_snr_est", allow_none=False)
    fft_mae = to_array("fft_mae", allow_none=False)
    gaussian_mae = to_array("gaussian_mae", allow_none=False)

    created_any = False

    # PSNR curves (only if GT exists for at least one sample)
    if not np.isnan(noisy_psnr).all() or not np.isnan(fft_psnr).all() or not np.isnan(gaussian_psnr).all():
        fig = plt.figure(figsize=(max(8, len(image_names) * 0.4), 4.5))
        plt.plot(indices, noisy_psnr, marker="o", label="Noisy (PSNR)", linewidth=1.5)
        plt.plot(indices, gaussian_psnr, marker="o", label="Gaussian PSNR", linewidth=1.5)
        plt.plot(indices, fft_psnr, marker="o", label="FFT PSNR", linewidth=1.5)
        plt.title("PSNR per Image")
        plt.xlabel("Image Index")
        plt.ylabel("PSNR (dB)")
        plt.xticks(indices, image_names, rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "psnr_curve.png", dpi=200)
        created_any = True

    # Estimated SNR curves
    if not np.isnan(fft_snr).all() or not np.isnan(gaussian_snr).all():
        fig = plt.figure(figsize=(max(8, len(image_names) * 0.4), 4.5))
        plt.plot(indices, gaussian_snr, marker="o", label="Gaussian est. SNR", linewidth=1.5)
        plt.plot(indices, fft_snr, marker="o", label="FFT est. SNR", linewidth=1.5)
        plt.title("Estimated SNR per Image (higher is better)")
        plt.xlabel("Image Index")
        plt.ylabel("Estimated SNR (dB)")
        plt.xticks(indices, image_names, rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "snr_curve.png", dpi=200)
        created_any = True

    # MAE curves (difference to noisy input)
    if not np.isnan(fft_mae).all() or not np.isnan(gaussian_mae).all():
        fig = plt.figure(figsize=(max(8, len(image_names) * 0.4), 4.5))
        plt.plot(indices, np.zeros_like(indices), label="Noisy baseline (0)", linestyle="--", color="gray")
        plt.plot(indices, gaussian_mae, marker="o", label="Gaussian MAE", linewidth=1.5)
        plt.plot(indices, fft_mae, marker="o", label="FFT MAE", linewidth=1.5)
        plt.title("Average |Noisy - Output| per Image")
        plt.xlabel("Image Index")
        plt.ylabel("MAE")
        plt.xticks(indices, image_names, rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "mae_curve.png", dpi=200)
        created_any = True

    if created_any:
        plt.show()
def build_model(device: torch.device) -> FFTDenoiser:
    """Instantiate the FFTDenoiser defined in learnable_denoising_v2 on the target device."""
    model = FFTDenoiser().to(device)
    return model


def load_checkpoint(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    """Load a checkpoint that stores either a raw state_dict or a dict containing it."""
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    model.load_state_dict(state)


def guess_gt_path(noisy_path: Path, gt_dir: Optional[Path]) -> Optional[Path]:
    """Try to locate a ground-truth file for the provided noisy image."""
    if gt_dir is not None:
        candidate = gt_dir / noisy_path.name
        if candidate.exists():
            return candidate

    replacements = [
        ("NOISY", "GT"),
        ("Noisy", "GT"),
        ("noisy", "gt"),
        ("_input", "_target"),
        ("_INPUT", "_GT"),
    ]

    for old, new in replacements:
        if old in noisy_path.name:
            candidate = noisy_path.with_name(noisy_path.name.replace(old, new))
            if candidate.exists():
                return candidate

    return None


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is not None and not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    resize_hw = None if args.no_resize else tuple(args.resize)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    if args.gaussian_ksize <= 0 or args.gaussian_ksize % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer.")
    if args.gaussian_sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    model = build_model(device)
    load_checkpoint(model, model_path, device)
    model.eval()

    image_paths = collect_images(input_dir)

    metrics = []
    print(f"Processing {len(image_paths)} image(s) on {device}...")

    for img_path in tqdm(image_paths, desc="Denoising", unit="img"):
        noisy_tensor, orig_size = load_image_tensor(img_path, resize_hw)
        noisy_tensor = noisy_tensor.to(device)

        with torch.no_grad():
            denoised_tensor = model(noisy_tensor).clamp(0.0, 1.0)
        gaussian_tensor = gaussian_denoise_tensor(noisy_tensor, args.gaussian_ksize, args.gaussian_sigma)
        fft_snr_est = estimate_snr(noisy_tensor, denoised_tensor)
        fft_mae = mean_abs_difference(noisy_tensor, denoised_tensor)
        gaussian_snr_est = estimate_snr(noisy_tensor, gaussian_tensor)
        gaussian_mae = mean_abs_difference(noisy_tensor, gaussian_tensor)

        dst_size = orig_size if resize_hw is not None else None
        result_img = tensor_to_image(denoised_tensor, dst_size)
        save_path = output_dir / f"{img_path.stem}_denoised.png"
        cv2.imwrite(str(save_path), result_img)
        gaussian_img = tensor_to_image(gaussian_tensor, dst_size)
        gaussian_path = output_dir / f"{img_path.stem}_gaussian.png"
        cv2.imwrite(str(gaussian_path), gaussian_img)

        psnr_noisy = None
        psnr_value = None
        psnr_gaussian = None
        gt_path = guess_gt_path(img_path, gt_dir)
        if gt_path is not None and gt_path.exists():
            gt_tensor, _ = load_image_tensor(gt_path, resize_hw)
            gt_tensor = gt_tensor.to(device)
            with torch.no_grad():
                psnr_noisy = calculate_psnr(noisy_tensor, gt_tensor).item()
                psnr_value = calculate_psnr(denoised_tensor, gt_tensor).item()
                psnr_gaussian = calculate_psnr(gaussian_tensor, gt_tensor).item()

        metrics.append(
            {
                "filename": img_path.name,
                "noisy_psnr": psnr_noisy,
                "fft_psnr": psnr_value,
                "fft_snr_est": fft_snr_est,
                "fft_mae": fft_mae,
                "fft_file": save_path.name,
                "gaussian_psnr": psnr_gaussian,
                "gaussian_snr_est": gaussian_snr_est,
                "gaussian_mae": gaussian_mae,
                "gaussian_file": gaussian_path.name,
            }
        )

        fft_msg = (
            f"FFT PSNR={psnr_value:.2f} dB"
            if psnr_value is not None
            else "FFT PSNR=N/A"
        )
        fft_msg += f", SNR={fft_snr_est:.2f} dB, MAE={fft_mae:.4f}"
        gauss_msg = (
            f"Gaussian PSNR={psnr_gaussian:.2f} dB"
            if psnr_gaussian is not None
            else "Gaussian PSNR=N/A"
        )
        gauss_msg += f", SNR={gaussian_snr_est:.2f} dB, MAE={gaussian_mae:.4f}"
        tqdm.write(f"{img_path.name}: {fft_msg}, {gauss_msg} -> {save_path}, {gaussian_path}")

    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "filename\tnoisy_psnr_dB\tfft_output\tfft_psnr_dB\tfft_snr_dB\tfft_mae\t"
            "gaussian_output\tgaussian_psnr_dB\tgaussian_snr_dB\tgaussian_mae\n"
        )
        for item in metrics:
            noisy_psnr_str = f"{item['noisy_psnr']:.4f}" if item["noisy_psnr"] is not None else "N/A"
            fft_psnr_str = f"{item['fft_psnr']:.4f}" if item["fft_psnr"] is not None else "N/A"
            gaussian_psnr_str = (
                f"{item['gaussian_psnr']:.4f}" if item["gaussian_psnr"] is not None else "N/A"
            )
            f.write(
                f"{item['filename']}\t{noisy_psnr_str}\t{item['fft_file']}\t{fft_psnr_str}\t"
                f"{item['fft_snr_est']:.4f}\t{item['fft_mae']:.6f}\t"
                f"{item['gaussian_file']}\t{gaussian_psnr_str}\t"
                f"{item['gaussian_snr_est']:.4f}\t{item['gaussian_mae']:.6f}\n"
            )

    noisy_psnr_values = [item["noisy_psnr"] for item in metrics if item["noisy_psnr"] is not None]
    fft_psnr_values = [item["fft_psnr"] for item in metrics if item["fft_psnr"] is not None]
    gaussian_psnr_values = [
        item["gaussian_psnr"] for item in metrics if item["gaussian_psnr"] is not None
    ]
    fft_snr_values = [
        item["fft_snr_est"] for item in metrics if not math.isinf(item["fft_snr_est"])
    ]
    gaussian_snr_values = [
        item["gaussian_snr_est"]
        for item in metrics
        if not math.isinf(item["gaussian_snr_est"])
    ]
    fft_mae_values = [item["fft_mae"] for item in metrics]
    gaussian_mae_values = [item["gaussian_mae"] for item in metrics]
    fft_max_change = max(metrics, key=lambda x: x["fft_mae"], default=None)
    gaussian_max_change = max(metrics, key=lambda x: x["gaussian_mae"], default=None)

    if not noisy_psnr_values and not fft_psnr_values and not gaussian_psnr_values:
        print(
            "No ground-truth images were found. Denoised images are saved, "
            "but PSNR could not be computed."
        )
    else:
        if noisy_psnr_values:
            print(
                f"Average noisy-input PSNR over {len(noisy_psnr_values)} image(s): "
                f"{np.mean(noisy_psnr_values):.2f} dB"
            )
        if fft_psnr_values:
            print(
                f"Average FFT-denoiser PSNR over {len(fft_psnr_values)} image(s): "
                f"{np.mean(fft_psnr_values):.2f} dB"
            )
        if gaussian_psnr_values:
            print(
                f"Average Gaussian baseline PSNR over {len(gaussian_psnr_values)} image(s): "
                f"{np.mean(gaussian_psnr_values):.2f} dB"
            )
    if fft_snr_values:
        print(
            f"Average FFT estimated SNR over {len(fft_snr_values)} image(s): "
            f"{np.mean(fft_snr_values):.2f} dB"
        )
    if gaussian_snr_values:
        print(
            f"Average Gaussian estimated SNR over {len(gaussian_snr_values)} image(s): "
            f"{np.mean(gaussian_snr_values):.2f} dB"
        )
    if fft_mae_values:
        print(
            f"Average FFT |noisy-denoised| over {len(fft_mae_values)} image(s): "
            f"{np.mean(fft_mae_values):.4f}"
        )
    if gaussian_mae_values:
        print(
            f"Average Gaussian |noisy-denoised| over {len(gaussian_mae_values)} image(s): "
            f"{np.mean(gaussian_mae_values):.4f}"
        )
    if fft_max_change:
        print(
            f"Greatest FFT change: {fft_max_change['filename']} "
            f"(MAE={fft_max_change['fft_mae']:.4f}, est. SNR={fft_max_change['fft_snr_est']:.2f} dB)"
        )
    if gaussian_max_change:
        print(
            f"Greatest Gaussian change: {gaussian_max_change['filename']} "
            f"(MAE={gaussian_max_change['gaussian_mae']:.4f}, est. SNR={gaussian_max_change['gaussian_snr_est']:.2f} dB)"
        )
    print(f"Per-image metrics saved to {metrics_path}")

    if not args.no_plot:
        plot_metric_curves(metrics, output_dir)


if __name__ == "__main__":
    main()
