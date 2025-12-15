import os
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ================================
# 0. Utility helpers
# ================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ================================
# 1. FFT Conv2D (with bias)
# ================================
class FFTConv2d(nn.Module):
    """
    Simple convolution implemented with FFT:
    - Supports any H, W (>= kernel_size)
    - Equivalent to circular padding because convolution happens in frequency domain
    - Structurally similar to nn.Conv2d(in_ch, out_ch, ksize, padding='same') but running in frequency domain
    """

    def __init__(self, in_ch, out_ch, ksize):
        super().__init__()
        self.ksize = ksize
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Weight initialization for better stability
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, ksize, ksize) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        """
        x: (B, Cin, H, W)
        """
        B, C, H, W = x.shape
        k = self.ksize
        pad = k // 2

        # Use reflection padding to avoid circular artifacts introduced by frequency-domain conv
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        H_pad, W_pad = x_pad.shape[-2:]

        # Pad kernel to the same spatial size as the padded input
        pad_h = H_pad - k
        pad_w = W_pad - k
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Padded input size (H={H_pad}, W={W_pad}) must be >= kernel size (k={k})"
            )

        w = F.pad(self.weight, (0, pad_w, 0, pad_h))

        # Circshift the kernel to the center to avoid phase shift
        w = torch.roll(w, shifts=(-k // 2, -k // 2), dims=(-2, -1))

        # ------ FFT ------
        Xf = torch.fft.fft2(x_pad)    # (B, Cin, H_pad, W_pad)
        Wf = torch.fft.fft2(w)        # (Cout, Cin, H_pad, W_pad)

        # ------ Proper channel-wise convolution via einsum ------
        # b: batch, i: in_ch, o: out_ch, h,w: spatial
        Yf = torch.einsum("bihw,oihw->bohw", Xf, Wf)

        # ------ iFFT ------
        y = torch.fft.ifft2(Yf).real  # (B, Cout, H_pad, W_pad)

        # Add bias
        y = y + self.bias.view(1, -1, 1, 1)

        # Crop the padding to restore the original spatial size
        y = y[..., pad:pad + H, pad:pad + W]

        return y


# ================================
# 2. Denoising Network
# ================================
class FFTResidualBlock(nn.Module):
    """Two-layer FFT convolution residual block that boosts receptive field and capacity."""

    def __init__(self, channels, ksize=5):
        super().__init__()
        self.conv1 = FFTConv2d(channels, channels, ksize)
        self.conv2 = FFTConv2d(channels, channels, ksize)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class FFTDenoiser(nn.Module):
    """
    Deeper FFT convolutional network that predicts noise and subtracts it residually for more stable training.
    """

    def __init__(self, base_channels=32, num_blocks=2, ksize=5):
        super().__init__()
        self.head = FFTConv2d(3, base_channels, ksize)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([
            FFTResidualBlock(base_channels, ksize) for _ in range(num_blocks)
        ])
        self.tail = FFTConv2d(base_channels, 3, ksize)

    def forward(self, x):
        feat = self.relu(self.head(x))
        for block in self.blocks:
            feat = block(feat)
        noise = self.tail(feat)
        # Predict noise and subtract it from the input, similar to the residual structure in DnCNN
        return x - noise


# ================================
# 3. SIDD Dataset (flatten all patches)
# ================================
class SIDDDataset(Dataset):
    """
    The data_root directory should look like:
      Data/
        0001/
          NOISY_SRGB_010.PNG
          NOISY_SRGB_011.PNG
          ...
          GT_SRGB_010.PNG
          GT_SRGB_011.PNG
          ...
        0002/
        ...

    All (NOISY, GT) patch pairs inside each folder are expanded into samples.
    """

    def __init__(self, data_root, transform=None, is_train=True, target_size=(256, 256)):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.target_size = target_size
        self.is_train = is_train

        all_groups = sorted([
            g for g in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, g))
        ])

        # Simple split: first 5 groups go to validation, the rest go to training
        if is_train:
            groups = all_groups[5:]
        else:
            groups = all_groups[:5]

        self.pairs = []
        for g in groups:
            group_path = os.path.join(data_root, g)
            files = os.listdir(group_path)

            noisy_files = sorted([f for f in files if "NOISY_SRGB" in f])
            clean_files = sorted([f for f in files if "GT_SRGB" in f])

            # Guard against mismatched counts
            n = min(len(noisy_files), len(clean_files))
            for i in range(n):
                noisy_path = os.path.join(group_path, noisy_files[i])
                clean_path = os.path.join(group_path, clean_files[i])
                self.pairs.append((noisy_path, clean_path))

        print(f"{'Train' if is_train else 'Val'} dataset size: {len(self.pairs)} samples")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        noisy = cv2.imread(noisy_path)
        clean = cv2.imread(clean_path)

        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        # Resize to a consistent spatial size
        noisy = cv2.resize(noisy, self.target_size, interpolation=cv2.INTER_CUBIC)
        clean = cv2.resize(clean, self.target_size, interpolation=cv2.INTER_CUBIC)

        noisy = noisy.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0

        # Simple data augmentation (training only)
        if self.is_train:
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=1).copy()
                clean = np.flip(clean, axis=1).copy()
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=0).copy()
                clean = np.flip(clean, axis=0).copy()

        if self.transform is not None:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        else:
            # Default HWC->CHW
            noisy = torch.from_numpy(noisy).permute(2, 0, 1)
            clean = torch.from_numpy(clean).permute(2, 0, 1)

        return noisy, clean


# ================================
# 4. PSNR (more numerically stable)
# ================================
def calculate_psnr(pred, target, max_val=1.0):
    """
    pred, target: (B, C, H, W) with values roughly in [0,1]
    Higher PSNR is better; typically >30 dB indicates good quality.
    """
    # Clamp to [0, max_val] to avoid overflow that makes mse > 1
    pred = torch.clamp(pred, 0.0, max_val)
    target = torch.clamp(target, 0.0, max_val)

    mse = torch.mean((pred - target) ** 2)

    if mse <= 1e-10:
        # Provide an upper bound when the output is extremely close to GT
        return torch.tensor(100.0, device=pred.device)

    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr


# ================================
# 5. Training
# ================================
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    output_dir="fft_results",
):

    device = get_device()
    print("Using device:", device)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_loss_list = []
    train_psnr_list = []
    val_psnr_list = []

    best_val_psnr = -1e9
    best_model_path = os.path.join(output_dir, "best_model.pth")

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} [Train]")

        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(noisy)
                loss = criterion(out, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                psnr_batch = calculate_psnr(out.detach(), clean)

            train_loss += loss.item()
            train_psnr += psnr_batch.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr_batch.item():.2f}")

        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)

        # ---------- Validation ----------
        model.eval()
        val_psnr = 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                psnr_batch = calculate_psnr(out, clean)
                val_psnr += psnr_batch.item()

        val_psnr /= len(val_loader)

        train_loss_list.append(train_loss)
        train_psnr_list.append(train_psnr)
        val_psnr_list.append(val_psnr)

        print(
            f"\nEpoch {ep+1}: "
            f"loss={train_loss:.4f}, train_psnr={train_psnr:.2f} dB, val_psnr={val_psnr:.2f} dB\n"
        )

        # Save the best-performing model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best val_psnr: {best_val_psnr:.2f} dB, model saved to {best_model_path}")

    return model, train_loss_list, train_psnr_list, val_psnr_list


# ================================
# 6. Main
# ================================
def main():
    set_seed(42)

    output_dir = "fft_results_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Keep only ToTensor since inputs are already float values in [0,1]
    transform = transforms.ToTensor()

    # Replace this with your own dataset path
    data_root = "data/SIDD_Small_sRGB_Only/Data"

    train_data = SIDDDataset(data_root, transform=transform, is_train=True)
    val_data = SIDDDataset(data_root, transform=transform, is_train=False)

    device = get_device()
    # Use a smaller batch on Apple Silicon to avoid running out of memory
    batch_size = 64 if device.type == "cuda" else 16

    # Configure DataLoader per device (single-threaded on MPS/CPU is more stable)
    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = FFTDenoiser()

    model, loss_list, train_psnr_list, val_psnr_list = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=1e-3,
        output_dir=output_dir,
    )

    # ----- Plot -----
    epochs = np.arange(1, len(loss_list) + 1)

    plt.figure()
    plt.plot(epochs, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_psnr_list, label="Train")
    plt.plot(epochs, val_psnr_list, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "psnr.png"))
    plt.close()

    print("Training complete. Curves saved to", output_dir)


if __name__ == "__main__":
    main()
