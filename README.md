# Image Deraining Using UNet-GAN

Removes rain streaks from images using a UNet Generator + PatchGAN Discriminator, trained on Rain200H dataset.

## Architecture
- **Generator:** UNet encoder-decoder with skip connections (512×512 → 16×16 bottleneck → 512×512)
- **Discriminator:** PatchGAN producing a 62×62 patch-realness grid

## Loss Functions
Composite generator loss with weights: L1 (1.0) + SSIM (0.1) + Perceptual/VGG (0.1) + Adversarial (0.01)

## Training
- Adam optimizer, lr=2×10⁻⁴, 200 epochs, batch size 8
- Best checkpoint at epoch 190 (early stopping on PSNR)

## Results on Rain200H

| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| DerainNet | 26.11 | 0.792 |
| JORDER | 22.05 | 0.727 |
| SEMI | 16.56 | 0.486 |
| **Our UNet-GAN** | **24.73** | **0.8065** |

## Images
![Uploading result_4.png…]()
![Uploading result_3.png…]()
![Uploading result_2.png…]()
![Uploading result_1.png…]()


> Best SSIM and 2nd best PSNR among all compared models.
