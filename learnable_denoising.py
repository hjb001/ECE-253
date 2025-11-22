import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# Define the FFT-based convolution layer
class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FFTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize learnable filter kernel
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.shape
        
        # Pad input to handle boundary effects
        x_padded = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        
        # Compute FFT of input
        x_fft = torch.fft.fftn(x_padded, dim=(-2, -1))
        
        # Create filter in frequency domain
        # Create a tensor for the padded weights with the same size as the padded input
        weight_padded = torch.zeros(batch_size, self.out_channels, in_channels, 
                                   x_padded.shape[-2], x_padded.shape[-1], device=x.device)
        
        # Copy the weights for each input channel and output channel
        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(in_channels):
                    # Place the kernel in the top-left corner
                    weight_padded[b, o, i, :self.kernel_size, :self.kernel_size] = self.weight[o, i, :, :]
                    
        # Move weight to the center (for proper convolution)
        weight_padded = torch.roll(weight_padded, shifts=(-self.kernel_size//2, -self.kernel_size//2), dims=(-2, -1))
        
        # Compute FFT of weight for each output channel
        weight_fft = torch.fft.fftn(weight_padded, dim=(-2, -1))
        
        # Expand x_fft to match the dimensions for element-wise multiplication
        # x_fft: (batch_size, in_channels, height, width)
        # We need to compute the convolution sum over input channels for each output channel
        x_fft_expanded = x_fft.unsqueeze(1)  # (batch_size, 1, in_channels, height, width)
        
        # Element-wise multiplication in frequency domain and sum over input channels
        # This simulates the convolution operation
        output_fft = torch.sum(x_fft_expanded * weight_fft, dim=2)  # Sum over input channels
        
        # Inverse FFT to get result in spatial domain
        output = torch.fft.ifftn(output_fft, dim=(-2, -1)).real
        
        # Crop to original size
        output = output[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return output

# Simple denoising network with FFT-based convolution
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

# Custom dataset for SIDD data
class SIDDDataset(Dataset):
    def __init__(self, data_root, transform=None, is_train=True, target_size=(256, 256)):
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train
        self.target_size = target_size
        
        # Get all groups
        self.groups = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        
        # Split into train/test - first 5 for test, rest for training
        if is_train:
            self.groups = self.groups[5:]  # Use all except first 5 for training
        else:
            self.groups = self.groups[:5]  # Use first 5 for testing
            
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        group_path = os.path.join(self.data_root, self.groups[idx])
        files = os.listdir(group_path)
        
        clean_file = None
        noisy_file = None
        
        for f in files:
            if 'GT_SRGB' in f and f.endswith('.PNG'):
                clean_file = f
            elif 'NOISY_SRGB' in f and f.endswith('.PNG'):
                noisy_file = f
                
        clean_path = os.path.join(group_path, clean_file)
        noisy_path = os.path.join(group_path, noisy_file)
        
        # Load images
        clean_img = cv2.imread(clean_path)
        noisy_img = cv2.imread(noisy_path)
        
        # Convert BGR to RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        
        # Resize images to target size
        clean_img = cv2.resize(clean_img, self.target_size)
        noisy_img = cv2.resize(noisy_img, self.target_size)
        
        # Convert to float and normalize
        clean_img = clean_img.astype(np.float32) / 255.0
        noisy_img = noisy_img.astype(np.float32) / 255.0
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
            
        return noisy_img, clean_img

# PSNR calculation function
def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    # Check for Metal Performance Shaders (MPS) support on macOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        
        # Use tqdm for progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for noisy_imgs, clean_imgs in train_progress:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_psnr += calculate_psnr(outputs, clean_imgs).item()
            
            # Update progress bar
            train_progress.set_postfix({
                'Loss': loss.item(),
                'PSNR': calculate_psnr(outputs, clean_imgs).item()
            })
            
        # Validation
        model.eval()
        val_psnr = 0.0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_progress:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                val_psnr += calculate_psnr(outputs, clean_imgs).item()
                val_progress.set_postfix({
                    'PSNR': calculate_psnr(outputs, clean_imgs).item()
                })
                
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)
        val_psnr /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB, Val PSNR: {val_psnr:.2f} dB')
        
    return model

# Main execution
def main():
    # Create output directory
    output_dir = "fft_denoise_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load dataset
    data_root = "data/SIDD_Small_sRGB_Only/Data"
    
    # Create datasets with consistent image size
    train_dataset = SIDDDataset(data_root, transform=transform, is_train=True, target_size=(256, 256))
    test_dataset = SIDDDataset(data_root, transform=transform, is_train=False, target_size=(256, 256))
    
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create model
    model = FFTDenoiser()
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
    
    # Test model and save results
    # Check for Metal Performance Shaders (MPS) support on macOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) backend for testing")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend for testing")
    else:
        device = torch.device("cpu")
        print("Using CPU backend for testing")
    
    model.to(device)
    model.eval()
    
    psnr_noisy_values = []
    psnr_denoised_values = []
    
    print("Testing and saving results...")
    with torch.no_grad():
        # Add progress bar for testing
        test_progress = tqdm(test_loader, desc="Testing")
        for i, (noisy_imgs, clean_imgs) in enumerate(test_progress):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Denoise
            denoised_imgs = model(noisy_imgs)
            
            # Calculate PSNR
            psnr_noisy = calculate_psnr(noisy_imgs, clean_imgs)
            psnr_denoised = calculate_psnr(denoised_imgs, clean_imgs)
            
            psnr_noisy_values.append(psnr_noisy.item())
            psnr_denoised_values.append(psnr_denoised.item())
            
            test_progress.set_postfix({
                'Noisy PSNR': psnr_noisy.item(),
                'Denoised PSNR': psnr_denoised.item()
            })
            
            # Save denoised image
            denoised_img = denoised_imgs[0].cpu().numpy().transpose(1, 2, 0)
            denoised_img = np.clip(denoised_img, 0, 1)
            denoised_img = (denoised_img * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            denoised_img_bgr = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/denoised_{i}.png", denoised_img_bgr)
    
    # Calculate average PSNR values
    avg_psnr_noisy = np.mean(psnr_noisy_values)
    avg_psnr_denoised = np.mean(psnr_denoised_values)
    
    print(f"\nProcessing completed! Saved {len(test_dataset)} denoised images.")
    print(f"Average PSNR of noisy images: {avg_psnr_noisy:.2f} dB")
    print(f"Average PSNR of denoised images: {avg_psnr_denoised:.2f} dB")
    print(f"Improvement: {avg_psnr_denoised - avg_psnr_noisy:.2f} dB")

if __name__ == "__main__":
    main()