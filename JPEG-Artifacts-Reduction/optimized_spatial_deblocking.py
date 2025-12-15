import os
import cv2
import numpy as np


def detect_block_boundaries(img, block_size=8):
    h, w = img.shape[:2]
    boundary_strength = np.zeros((h, w), dtype=np.float32)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if j + block_size < w:
                boundary_x = j + block_size
                left_val = gradient_mag[i:min(i+block_size, h), boundary_x-1]
                right_val = gradient_mag[i:min(i+block_size, h), boundary_x]
                discontinuity = np.abs(right_val - left_val)
                discontinuity_2d = np.repeat(discontinuity[:, None], 2, axis=1)
                boundary_strength[i:min(i+block_size, h), boundary_x-1:boundary_x+1] += discontinuity_2d
            
            if i + block_size < h:
                boundary_y = i + block_size
                top_val = gradient_mag[boundary_y-1, j:min(j+block_size, w)]
                bottom_val = gradient_mag[boundary_y, j:min(j+block_size, w)]
                discontinuity = np.abs(bottom_val - top_val)
                discontinuity_2d = np.repeat(discontinuity[None, :], 2, axis=0)
                boundary_strength[boundary_y-1:boundary_y+1, j:min(j+block_size, w)] += discontinuity_2d
    
    if boundary_strength.max() > 0:
        boundary_strength = boundary_strength / boundary_strength.max()
    
    return boundary_strength


def optimized_spatial_deblocking(img,
                                  block_size=8,
                                  boundary_width=2,
                                  base_blur_kernel=(3, 3),
                                  adaptive_strength=0.3):
    h, w = img.shape[:2]
    
    boundary_strength = detect_block_boundaries(img, block_size)
    
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    
    for x in range(0, w, block_size):
        x_start = max(0, x - boundary_width)
        x_end = min(w, x + boundary_width + 1)
        boundary_mask[:, x_start:x_end] = 1
    
    for y in range(0, h, block_size):
        y_start = max(0, y - boundary_width)
        y_end = min(h, y + boundary_width + 1)
        boundary_mask[y_start:y_end, :] = 1
    
    blur_weight = boundary_mask.astype(np.float32) * (
        0.5 + 0.5 * boundary_strength * adaptive_strength
    )
    blur_weight = np.clip(blur_weight, 0, 1)
    
    blurred = cv2.GaussianBlur(img, base_blur_kernel, 0)
    
    if len(img.shape) == 3:
        blur_weight = blur_weight[:, :, None]
    
    out = img.astype(np.float32) * (1 - blur_weight) + blurred.astype(np.float32) * blur_weight
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out


def process_folder(input_dir, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        
        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Failed to read: {in_path}")
            continue
        
        deblocked = optimized_spatial_deblocking(img, **kwargs)
        
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, deblocked)
        print(f"Optimized deblocking done: {fname}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10")
    output_dir = os.path.join(root_dir, "data", "BSDS500", "JPEG_Q10_optimized")

    process_folder(
        input_dir,
        output_dir,
        block_size=8,
        boundary_width=2,
        base_blur_kernel=(3, 3),
        adaptive_strength=0.4,
    )

    print("All optimized deblocking finished.")

