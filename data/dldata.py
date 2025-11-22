import kagglehub
import shutil
import os

import os
os.environ["KAGGLEHUB_CACHE"] = "I:/kagglehub_cache"
# 下载数据集
path = kagglehub.dataset_download("rajat95gupta/smartphone-image-denoising-dataset")
print("Downloaded to:", path)

# 目标目录（当前目录）
target_dir = "./"  # 或者 os.getcwd()

# 将下载好的内容复制到当前目录
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)

    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset moved to current folder.")
