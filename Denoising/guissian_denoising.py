import h5py
import numpy as np
import cv2
import os

# -------------------------
# 1. 读取 SIDD GT 图像
# -------------------------
def load_sidd_gt(mat_path):
    f = h5py.File(mat_path, 'r')
    
    # 检查文件中可用的数据集键
    print("Available keys in the mat file:", list(f.keys()))
    
    # 对于BenchmarkNoisyBlocksRaw.mat，数据结构可能不同
    if 'BenchmarkNoisyBlocksRaw' in f:
        gt = np.array(f['BenchmarkNoisyBlocksRaw'])
        # 根据实际情况调整维度转换
        if len(gt.shape) == 4 and gt.shape[0] == 1:
            # 如果是 (1, N, H, W) 形状，则压缩第一维并转置
            gt = np.squeeze(gt, axis=0)  # 移除第一维变成 (N, H, W)
            # 添加通道维度变为 (N, H, W, 1)
            gt = np.expand_dims(gt, axis=-1)
        elif len(gt.shape) == 4:
            # 如果是 (N, 3, H, W) 或其他4D格式
            gt = np.transpose(gt, (0, 2, 3, 1))  # 转为 N H W C
    else:
        # 原来的逻辑用于兼容其他格式
        gt = np.array(f['GT'])
        gt = np.transpose(gt, (0, 2, 3, 1))  # 转为 N H W C
        
    return gt

# -------------------------
# 2. 加高斯噪声
# -------------------------
def add_gaussian_noise(img, sigma=25):
    noise = np.random.randn(*img.shape) * (sigma / 255.)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy

# -------------------------
# 3. 高斯去噪
# -------------------------
def gaussian_denoise(noisy, ksize=5, sigma=1.2):
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    denoised = cv2.GaussianBlur(noisy_uint8, (ksize, ksize), sigmaX=sigma)
    return denoised

# -------------------------
# 4. 计算PSNR指标
# -------------------------
def calculate_psnr(img1, img2):
    """
    计算两幅图像之间的PSNR值
    """
    # 计算均方误差(MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # 如果MSE为0，说明两幅图像完全相同，返回无穷大
    if mse == 0:
        return float('inf')
    
    # 计算PSNR值
    # MAX_I = 1.0 因为我们的图像已经归一化到了[0,1]范围
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

# =========================
# 新增：读取SIDD_Small数据集函数
# =========================
def load_sidd_small_data(data_root, num_groups=5):
    """
    从SIDD_Small数据集中加载前num_groups组数据
    """
    groups = sorted(os.listdir(data_root))[:num_groups]
    clean_images = []
    noisy_images = []
    
    for group in groups:
        group_path = os.path.join(data_root, group)
        if not os.path.isdir(group_path):
            continue
            
        # 查找干净图像和噪声图像
        files = os.listdir(group_path)
        clean_file = None
        noisy_file = None
        
        for f in files:
            if 'GT_SRGB' in f and f.endswith('.PNG'):
                clean_file = f
            elif 'NOISY_SRGB' in f and f.endswith('.PNG'):
                noisy_file = f
        
        if clean_file and noisy_file:
            clean_path = os.path.join(group_path, clean_file)
            noisy_path = os.path.join(group_path, noisy_file)
            
            clean_img = cv2.imread(clean_path)
            noisy_img = cv2.imread(noisy_path)
            
            if clean_img is not None and noisy_img is not None:
                # 转换为浮点数并归一化到[0,1]范围
                clean_img = clean_img.astype(np.float32) / 255.0
                noisy_img = noisy_img.astype(np.float32) / 255.0
                
                # 转换颜色通道顺序从BGR到RGB
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
                
                clean_images.append(clean_img)
                noisy_images.append(noisy_img)
    
    return clean_images, noisy_images

# =========================
# 主程序
# =========================

output_dir = "gaussian_denoise_results"
os.makedirs(output_dir, exist_ok=True)

# 加载SIDD_Small数据集的前5组数据
data_root = "data/SIDD_Small_sRGB_Only/Data"
clean_images, noisy_images = load_sidd_small_data(data_root, num_groups=5)
print(f"Loaded {len(clean_images)} pairs of images")

# 存储PSNR值用于统计
psnr_noisy_values = []
psnr_denoised_values = []

# 只处理前5组图像
num_images_to_process = len(clean_images)

for i in range(num_images_to_process):
    clean = clean_images[i]
    noisy = noisy_images[i]
    
    # 高斯滤波去噪
    denoised = gaussian_denoise(noisy, ksize=7, sigma=2)
    
    # 将去噪后的图像转换回[0,1]范围的浮点数格式以便计算PSNR
    denoised_float = denoised.astype(np.float32) / 255.0
    
    # 计算PSNR值
    psnr_noisy = calculate_psnr(clean, noisy)
    psnr_denoised = calculate_psnr(clean, denoised_float)
    
    psnr_noisy_values.append(psnr_noisy)
    psnr_denoised_values.append(psnr_denoised)
    
    print(f"Image {i}: PSNR of noisy image: {psnr_noisy:.2f} dB, PSNR of denoised image: {psnr_denoised:.2f} dB")
    
    
    # 仅保存去噪后的图像（转换回BGR用于OpenCV保存）
    denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/denoised_{i}.png", denoised_bgr)

# 计算平均PSNR值
avg_psnr_noisy = np.mean(psnr_noisy_values)
avg_psnr_denoised = np.mean(psnr_denoised_values)

print(f"\n处理完成！已保存 {num_images_to_process} 张去噪后的图像。")
print(f"Average PSNR of noisy images: {avg_psnr_noisy:.2f} dB")
print(f"Average PSNR of denoised images: {avg_psnr_denoised:.2f} dB")
print(f"Improvement: {avg_psnr_denoised - avg_psnr_noisy:.2f} dB")