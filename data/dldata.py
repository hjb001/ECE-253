import kagglehub
import shutil
import os

import os
os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), "kagglehub_cache")
# Download dataset
path = kagglehub.dataset_download("rajat95gupta/smartphone-image-denoising-dataset")
print("Downloaded to:", path)

# Target directory (current directory)
target_dir = "./"  # or os.getcwd()

# Copy the downloaded content to the current directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)

    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset moved to current folder.")
