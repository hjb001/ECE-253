import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# FFT-based Conv2d Layer
# =========================
class FFTConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ksize, ksize))
        self.ksize = ksize

    def forward(self, x):
        B = int(x.size(0))
        C = int(x.size(1))
        H = int(x.size(2))
        W = int(x.size(3))

        pad_h = H - self.ksize
        pad_w = W - self.ksize

        weight_padded = F.pad(self.weight, (0, pad_w, 0, pad_h))

        X_f = torch.fft.rfftn(x, dim=(-2, -1))
        W_f = torch.fft.rfftn(weight_padded, dim=(-2, -1))

        Y_f = torch.sum(X_f.unsqueeze(1) * W_f.unsqueeze(0), dim=2)

        y = torch.fft.irfftn(Y_f, s=(H, W), dim=(-2, -1))
        return y


# =========================
# Denoising Network
# =========================
class FFTDenoiser(nn.Module):
    def __init__(self):
        super(FFTDenoiser, self).__init__()
        self.conv1 = FFTConv2d(3, 32, 5)
        self.conv2 = FFTConv2d(32, 32, 5)
        self.conv3 = FFTConv2d(32, 3, 5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# =========================
# SIDD Dataset
# =========================
class SIDDDataset(Dataset):
    def __init__(self, data_root, transform=None, is_train=True, target_size=(256, 256)):
        self.data_root = data_root
        self.transform = transform
        self.target_size = target_size

        self.groups = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])

        if is_train:
            self.groups = self.groups[5:]
        else:
            self.groups = self.groups[:5]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        path = os.path.join(self.data_root, self.groups[idx])
        files = os.listdir(path)

        clean_file = [f for f in files if "GT_SRGB" in f][0]
        noisy_file = [f for f in files if "NOISY_SRGB" in f][0]

        clean = cv2.imread(os.path.join(path, clean_file))
        noisy = cv2.imread(os.path.join(path, noisy_file))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        clean = cv2.resize(clean, self.target_size).astype(np.float32) / 255.0
        noisy = cv2.resize(noisy, self.target_size).astype(np.float32) / 255.0

        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)

        return noisy, clean


# =========================
# PSNR
# =========================
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse)


# =========================
# Training
# =========================
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):

    # ===========================================
    # FORCE cpus
    # ===========================================
    device = torch.device("cuda")


    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用于绘图
    train_loss_list = []
    train_psnr_list = []
    val_psnr_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_psnr = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for noisy, clean in prog:
            noisy, clean = noisy.to(device), clean.to(device)

            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += calculate_psnr(output, clean).item()

            prog.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)

        train_loss_list.append(train_loss)
        train_psnr_list.append(train_psnr)

        # ========== Validation ==========
        model.eval()
        val_psnr = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                val_psnr += calculate_psnr(out, clean).item()

        val_psnr /= len(val_loader)
        val_psnr_list.append(val_psnr)

        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, train_psnr={train_psnr:.2f}, val_psnr={val_psnr:.2f}")

    return model, train_loss_list, train_psnr_list, val_psnr_list


# =========================
# Main
# =========================
def main():
    output_dir = "fft_denoise_results"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.ToTensor()
    data_root = "data/SIDD_Small_sRGB_Only/Data"

    train_dataset = SIDDDataset(data_root, transform=transform, is_train=True)
    test_dataset = SIDDDataset(data_root, transform=transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FFTDenoiser()

    print("Starting Training on GPU...")
    model, loss_curve, train_psnr_curve, val_psnr_curve = train_model(
        model, train_loader, test_loader, num_epochs=20, learning_rate=0.001
    )

    # =========================
    # 绘制 Loss 和 PSNR 曲线
    # =========================
    epochs = np.arange(1, len(loss_curve) + 1)

    # ---- Loss ----
    plt.figure()
    plt.plot(epochs, loss_curve, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))

    # ---- PSNR ----
    plt.figure()
    plt.plot(epochs, train_psnr_curve, label="Train PSNR")
    plt.plot(epochs, val_psnr_curve, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "psnr_curve.png"))

    print("Saved loss_curve.png and psnr_curve.png !")


if __name__ == "__main__":
    main()
