import os
import cv2
import numpy as np


def pad_to_multiple(img, block_size):
    h, w = img.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if pad_h == 0 and pad_w == 0:
        return img, (0, 0, 0, 0)

    padded = cv2.copyMakeBorder(
        img,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, (0, pad_h, 0, pad_w)


def unpad(img, pads):
    top, bottom, left, right = 0, pads[1], 0, pads[3]
    h, w = img.shape[:2]
    return img[
        top : h - bottom if bottom > 0 else h,
        left : w - right if right > 0 else w,
    ]


def dct_deblocking_y_channel(bgr_img, block_size=8, cutoff=4, alpha=0.5):
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y = y.astype(np.float32)

    y_padded, pads = pad_to_multiple(y, block_size)
    h, w = y_padded.shape

    y_out = np.zeros_like(y_padded)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = y_padded[i : i + block_size, j : j + block_size]
            dct_block = cv2.dct(block)
            for u in range(block_size):
                for v in range(block_size):
                    if (u + v) > cutoff:
                        dct_block[u, v] *= alpha
            idct_block = cv2.idct(dct_block)
            y_out[i : i + block_size, j : j + block_size] = idct_block

    y_out = unpad(y_out, pads)

    y_out = np.clip(y_out, 0, 255).astype(np.uint8)
    ycrcb_out = cv2.merge([y_out, cr, cb])
    bgr_out = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)

    return bgr_out


def process_folder_dct_deblocking(input_dir, output_dir, block_size=8, cutoff=4, alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Failed to read: {in_path}")
            continue

        deblocked = dct_deblocking_y_channel(
            img,
            block_size=block_size,
            cutoff=cutoff,
            alpha=alpha,
        )

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, deblocked)
        print(f"DCT deblocking done: {fname}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10")
    output_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10_dct")

    process_folder_dct_deblocking(
        input_dir,
        output_dir,
        block_size=8,
        cutoff=4,
        alpha=0.5,
    )

    print("All DCT deblocking finished.")
