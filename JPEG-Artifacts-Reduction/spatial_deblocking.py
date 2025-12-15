import os
import cv2
import numpy as np


def spatial_deblocking(img, block_size=8, boundary_width=2, blur_kernel=(3, 3)):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)

    mask = np.zeros((h, w), dtype=np.uint8)

    for x in range(0, w, block_size):
        x_start = max(0, x - boundary_width)
        x_end = min(w, x + boundary_width + 1)
        mask[:, x_start:x_end] = 1

    for y in range(0, h, block_size):
        y_start = max(0, y - boundary_width)
        y_end = min(h, y + boundary_width + 1)
        mask[y_start:y_end, :] = 1

    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)

    out = img.copy()
    out[mask_3c == 1] = blurred[mask_3c == 1]

    return out


def process_folder(input_dir, output_dir, block_size=8, boundary_width=2, blur_kernel=(3, 3)):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Failed to read: {in_path}")
            continue

        deblocked = spatial_deblocking(
            img,
            block_size=block_size,
            boundary_width=boundary_width,
            blur_kernel=blur_kernel,
        )

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, deblocked)
        print(f"Spatial deblocking done: {fname}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10")
    output_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10_spatial")

    process_folder(
        input_dir,
        output_dir,
        block_size=8,
        boundary_width=2,
        blur_kernel=(3, 3),
    )
    print("All spatial deblocking finished.")
