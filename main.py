import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
import torchvision.models as models
from torch.utils.data import random_split

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure

random_seed = 42
torch.manual_seed(random_seed)

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)

# PatchGan Discriminator for UNet Generator
# For 512*512
class DerainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Input image size [3, 512, 512]

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ) # Output image size of block1 [64, 256, 256]

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ) # Output image size of block2 [128, 128, 128]

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ) # Output image size of block3 [256, 64, 64]

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ) # Output image size of block4 [512, 63, 63]

        self.last = nn.Conv2d(512, 1, 4, 1, 1)
        # Output image size of last layer [1, 62, 62]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.last(x)

# UNet Generator
# For 512*512
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DerainGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16
        # Input image size [3, 512, 512]
        self.down1 = DownBlock(3, 64, use_norm=False) # Output image size of down1 [64, 256, 256] (512 -> 256)
        self.down2 = DownBlock(64, 128) # Output image size of down2 [128, 128, 128] (256 -> 128)
        self.down3 = DownBlock(128, 256) # Output image size of down3 [256, 64, 64] (128 -> 64)
        self.down4 = DownBlock(256, 512) # Output image size of down4 [512, 32, 32] (64 -> 32)
        self.down5 = DownBlock(512, 512) # Output image size of down5 [512, 16, 16] (32 -> 16)  (bottleneck produced here)

        # Decoder (upsampling)
        # input channels = bottleneck channels (512)
        self.up1 = UpBlock(512, 512) # Output image size of up1 [512, 32, 32] (16 -> 32)
        self.up2 = UpBlock(1024, 512) # Output image size of up2 [512, 64, 64] (after cat -> 1024 in -> 512 out, 32 -> 64)
        self.up3 = UpBlock(768, 256) # Output image size of up3 [256, 128, 128] (after cat with d3 -> 768 in -> 256 out, 64 -> 128)
        self.up4 = UpBlock(384, 128) # Output image size of up4 [128, 256, 256] (after cat with d2 -> 384 in -> 128 out, 128 -> 256)
        self.up5 = UpBlock(192, 64) # Output image size of up5 [64, 512, 512] (after cat with d1 -> 192 in -> 64 out, 256 -> 512)

        # final output conv (no BN, tanh)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ) # Output image size of final layer [3, 512, 512]

    def forward(self, x, output_size=None):
        # encode (keep features for skips)
        d1 = self.down1(x) # 256
        d2 = self.down2(d1) # 128
        d3 = self.down3(d2) # 64
        d4 = self.down4(d3) # 32
        d5 = self.down5(d4) # 16 (bottleneck)

        # decode + concat skips
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1) # cat => 1024

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1) # cat => 768

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1) # cat => 384

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1) # cat => 192

        u5 = self.up5(u4)

        out = self.final(u5)

        if output_size is not None:
          out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)

        return out

# Path to dataset directory
path = './drive/MyDrive/RnD Project/Rain/heavy'

# Directories
RAINY_DIR = path+'/train/rain'
CLEAN_DIR = path+'/train/norain'
OUTPUT_DIR_TRAIN = path+'/outputs/training'
OUTPUT_DIR_TEST = path+'/outputs/testing'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class RainDataset(Dataset):
    def __init__(self, rainy_dir, clean_dir):
        self.rainy_images = sorted(os.listdir(rainy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir

        self.input_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.original_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        rainy = Image.open(rainy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        original_size = clean.size
        rainy = self.input_transform(rainy)
        # clean = self.original_transform(clean)
        clean = self.input_transform(clean)

        return rainy, clean, original_size, self.clean_images[idx]

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(fake, real):
    # input to vgg must be [0,1] range (not [-1,1])
    f = (fake*0.5 + 0.5)
    r = (real*0.5 + 0.5)
    feat_f = vgg(f)
    feat_r = vgg(r)
    return torch.nn.functional.l1_loss(feat_f, feat_r)

# Checkpoints
def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }, path)
def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    return checkpoint['epoch']

def evaluate_val(gen, val_loader):
    gen.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for rainy, clean, _, _ in val_loader:
            rainy = rainy.to(device)
            clean = clean.to(device)
            fake = gen(rainy, output_size=(clean.size(2), clean.size(3)))

            fake  = (fake*0.5 + 0.5).clamp(0,1).cpu()
            clean = (clean*0.5 + 0.5).clamp(0,1).cpu()

            # convert B*C*H*W -> B*H*W*C numpy
            fake_np  = fake.permute(0,2,3,1).numpy()
            clean_np = clean.permute(0,2,3,1).numpy()

            # compute psnr/ssim for whole batch vectorized
            batch_psnr = [psnr(clean_np[i], fake_np[i], data_range=1.0) for i in range(fake_np.shape[0])]
            batch_ssim = [ssim(clean_np[i], fake_np[i], channel_axis=2, data_range=1.0) for i in range(fake_np.shape[0])]

            total_psnr += sum(batch_psnr)
            total_ssim += sum(batch_ssim)
            count += fake_np.shape[0]

    return total_psnr / count, total_ssim / count

# Init
gen = DerainGenerator().to(device)
disc = DerainDiscriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
g_optim = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optim = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Data
train_rain_dir = path+'/train/rain'
train_clean_dir = path+'/train/norain'

test_rain_dir = path+'/test/rain'
test_clean_dir = path+'/test/norain'

test_dataset = RainDataset(test_rain_dir, test_clean_dir)

full_train = RainDataset(train_rain_dir, train_clean_dir)
val_ratio = 0.1
val_size = int(len(full_train) * val_ratio)
train_size = len(full_train) - val_size

train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Train
print(f"Using device: {device}")
epochs = 200
start_epoch = 0
best_val_psnr = 0
G_losses = []
D_losses = []
Val_PSNR  = []
checkpoint_path = path+'/checkpoints/checkpoint.pth'
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(gen, disc, g_optim, d_optim, checkpoint_path)
    print(f"Resumed training from epoch {start_epoch}")
if os.path.exists(path+"/checkpoints/best.txt"):
  best_val_psnr = float(torch.load(path+"/checkpoints/best.txt", map_location=device, weights_only=False))
  print(f"Best PSNR score so far: {best_val_psnr}")

for epoch in range(start_epoch,epochs):
    gen.train()
    disc.train()

    for i, (rainy, clean, orig_size, filenames) in enumerate(train_loader):
        rainy = rainy.to(device)
        clean = clean.to(device)

        # Train Discriminator
        fake = gen(rainy, output_size=(clean.size(2), clean.size(3))).detach()
        real_out = disc(clean)
        fake_out = disc(fake)

        real_labels = torch.ones_like(real_out).to(device)
        fake_labels = torch.zeros_like(fake_out).to(device)

        d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Train Generator
        fake = gen(rainy, output_size=(clean.size(2), clean.size(3)))
        fake_out = disc(fake)
        recon_loss = F.l1_loss(fake, clean)
        ssim_val = structural_similarity_index_measure(fake, clean)
        l_ssim = 1 - ssim_val
        l_perc = perceptual_loss(fake, clean)
        g_loss = 0.01 * criterion(fake_out, real_labels) + recon_loss + 0.1 * l_ssim + 0.1 * l_perc
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save output images occasionally
        if i % 50 == 0:
            for j in range(fake.size(0)):
                img = fake[j].detach().cpu() * 0.5 + 0.5  # unnormalize
                vutils.save_image(img, os.path.join(OUTPUT_DIR_TRAIN, f"train_epoch{epoch+1}_step{i}_{filenames[j]}_output.png"))

    # Save checkpoint at best epoch
    val_psnr, val_ssim = evaluate_val(gen, val_loader)
    G_losses.append(g_loss.item()) # last batch G loss
    D_losses.append(d_loss.item()) # last batch D loss
    Val_PSNR.append(val_psnr)
    print(f"Val PSNR: {val_psnr:.3f}, Val SSIM: {val_ssim:.3f}")
    # save best checkpoint
    if val_psnr > best_val_psnr:
      best_val_psnr = val_psnr
      save_checkpoint(gen, disc, g_optim, d_optim, epoch+1, path=path+"/checkpoints/best_model.pth")
      torch.save(val_psnr,path+"/checkpoints/best.txt")
      print(">>> saved BEST checkpoint")
    save_checkpoint(gen, disc, g_optim, d_optim, epoch+1, path=checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Testing
if os.path.exists(path+"/checkpoints/best_model.pth"):
    best_epoch = load_checkpoint(gen, disc, g_optim, d_optim, path+"/checkpoints/best_model.pth")
    print(f"Model from best epoch {best_epoch}")
gen.eval()
total_psnr = 0
total_ssim = 0
count = 0

with torch.no_grad():
    for i, (rainy, clean, orig_sizes, filenames) in enumerate(test_loader):
        for j in range(rainy.size(0)):
            # Prepare input
            single_rainy = rainy[j].unsqueeze(0).to(device)
            single_clean = clean[j].unsqueeze(0).to(device)

            # Get original size (width, height), then to (height, width)
            orig_size = orig_sizes[0][j].item(), orig_sizes[1][j].item()
            orig_size_hw = (orig_size[1], orig_size[0])

            # Generate output
            output = gen(single_rainy, output_size=orig_size_hw)
            output_img = output[0].cpu() * 0.5 + 0.5 # unnormalize to [0, 1]
            clean_img = TF.resize(clean[j], size=orig_size_hw) # match size
            clean_img = clean_img.cpu() * 0.5 + 0.5 # unnormalize to [0, 1]

            # Convert to numpy (H, W, C) and float32
            output_np = output_img.permute(1, 2, 0).numpy()
            clean_np = clean_img.permute(1, 2, 0).numpy()

            # Clamp just in case
            output_np = np.clip(output_np, 0, 1)
            clean_np = np.clip(clean_np, 0, 1)

            # Compute metrics
            psnr_score = psnr(clean_np, output_np, data_range=1.0)
            ssim_score = ssim(clean_np, output_np, channel_axis=2, data_range=1.0)

            total_psnr += psnr_score
            total_ssim += ssim_score
            count += 1

            # Save image
            vutils.save_image(output_img, os.path.join(OUTPUT_DIR_TEST, f"test_{filenames[j]}_output.png"))

# Average metrics
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print(f"\nAverage PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")