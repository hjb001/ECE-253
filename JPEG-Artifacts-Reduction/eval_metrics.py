import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(ROOT_DIR, "data", "BSDS500", "GT")

METHODS = {
    "JPEG_Q10": os.path.join(ROOT_DIR, "data", "BSDS500", "JPEG_Q10"),
    "Spatial": os.path.join(ROOT_DIR, "data", "BSDS500", "JPEG_Q10_spatial"),
    "DCT": os.path.join(ROOT_DIR, "data", "BSDS500", "JPEG_Q10_dct"),
    "Optimized": os.path.join(ROOT_DIR, "data", "BSDS500", "JPEG_Q10_optimized"),
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    return sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)]
    )


def load_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return img_rgb


def compute_metrics(gt, pred):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
    return psnr, ssim


def main():
    gt_files = list_images(GT_DIR)
    print(f"Found {len(gt_files)} GT images.")

    results = {name: {"psnr": [], "ssim": []} for name in METHODS.keys()}

    for fname in gt_files:
        gt_path = os.path.join(GT_DIR, fname)
        gt_img = load_image(gt_path)

        h_gt, w_gt = gt_img.shape[:2]
        base_name = os.path.splitext(fname)[0]

        for method_name, method_dir in METHODS.items():
            pred_path = os.path.join(method_dir, fname)

            if not os.path.exists(pred_path):
                found = False
                for ext in VALID_EXTS:
                    alt_path = os.path.join(method_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        pred_path = alt_path
                        found = True
                        break

                if not found:
                    print(f"[Warning] {method_name} missing file for: {base_name}")
                    continue

            pred_img = load_image(pred_path)
            h_p, w_p = pred_img.shape[:2]

            if (h_p != h_gt) or (w_p != w_gt):
                pred_img = cv2.resize(
                    pred_img,
                    (w_gt, h_gt),
                    interpolation=cv2.INTER_CUBIC,
                )

            psnr, ssim = compute_metrics(gt_img, pred_img)
            results[method_name]["psnr"].append(psnr)
            results[method_name]["ssim"].append(ssim)

    print("\n====== Average Evaluation Results ======")
    for method_name, metrics in results.items():
        if len(metrics["psnr"]) == 0:
            print(f"{method_name}: no valid images, please check paths and filenames.")
            continue

        mean_psnr = np.mean(metrics["psnr"])
        mean_ssim = np.mean(metrics["ssim"])
        print(f"[{method_name}]  PSNR: {mean_psnr:.3f} dB   SSIM: {mean_ssim:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
