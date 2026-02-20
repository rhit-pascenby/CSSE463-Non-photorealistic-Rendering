# CSSE463 Non-photorealistic Rendering - Cartoon Style Transfer

A GAN-based deep learning system that transforms realistic photos into cartoon-style artwork. Final project for CSSE463 Image Recognition.

## Attribution

This project is based on the research paper and architecture from:
- **CTSS (Cartoon Style Transfer System)**: https://github.com/XiangGao1102/CTSS
- The original generator architecture (non-UNet version) follows the model design from the CTSS repository

### Dataset Sources

The training datasets used in this project are from:
- **Google Drive Dataset**: https://drive.google.com/drive/folders/1xtMNvpk7OonNbK-ZjINRSXyq0wYiAF40
  - Training photos (`train_photo/`)
  - DragonBall cartoon images (`DB/`)
  - The Wind Rises anime images (`TWR/`)
- This dataset is also referenced in the XiangGao CTSS GitHub repository

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Using the Web Interface](#using-the-web-interface)
- [Model Evaluation](#model-evaluation)
- [Architecture Options](#architecture-options)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)


### Install Dependencies

```bash
pip install torch torchvision pillow numpy tqdm
```

For web interface:
```bash
pip install flask
```

## Dataset Setup

### Organize Your Data

Create two directories with your training images:

```
CSSE463-Non-photorealistic-Rendering/
├── train_photo/          # Real photos (input style)
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── DB/                   # Cartoon/anime images (target style - DragonBall)
    ├── cartoon1.jpg
    ├── cartoon2.jpg
    └── ...
```

**Alternative target styles:**
- `TWR/` - The Wind Rises anime style
- `Both/` - Combined dataset (DragonBall + The Wind Rises)

### Dataset Requirements

- **Supported formats**: JPG, PNG, JPEG, BMP
- **Minimum images**: 500+ per directory (more is better)
- **Pairing**: Images do not need to correspond (unpaired training)
- **Resolution**: Any size (will be resized to 256x256 automatically)

### Using Pre-trained Models

Instead of training from scratch, you can download pre-trained checkpoints:

**Download Some Pre-trained Epochs:**
https://drive.google.com/drive/folders/14Mo-VA6WtvTjSsU9crXexvtMdvJoD3VA?usp=sharing

The download includes three trained model styles:
- **The_Wind_Rises_Epoch/** - Anime style based on "The Wind Rises"
- **Dragon_Ball_Epoch/** - Anime style based on DragonBall


After downloading, extract the checkpoint folders into your project root directory.



## Training the Model

### Basic Training Command

To start training with default settings:

```bash
python train.py
```

This will:
- Use the original generator architecture
- Train for 100 epochs
- Save checkpoints every 2 epochs to `checkpoints/`
- Generate sample outputs to `samples/`
- Create a training log file

### Training with U-Net Architecture

To train using the U-Net generator (better detail preservation):

```bash
python train.py --use-unet
```

### Resume Training from Checkpoint

To continue training from a saved checkpoint:

```bash
python train.py --checkpoint 50
```

Or specify the full checkpoint filename:

```bash
python train.py --checkpoint checkpoint_epoch_50.pth
```

To resume U-Net training:

```bash
python train.py --use-unet --checkpoint 50
```

### Configuring Training Parameters

Edit the config dictionary in [train.py](train.py#L495-L510):

```python
config = {
    'image_size': 256,              # Image resolution
    'batch_size': 4,                # Reduce to 2 or 1 if out of memory
    'num_epochs': 100,              # Total training epochs
    'lr_g': 0.0002,                 # Generator learning rate
    'lr_d': 0.0002,                 # Discriminator learning rate
    'lambda_contrastive': 1.0,      # Contrastive loss weight
    'lambda_identity': 5.0,         # Content preservation weight
    'lambda_saturation': 2.0,       # Color saturation weight
    'target_saturation': 0.7,       # Target saturation level
    'save_interval': 2,             # Save checkpoint frequency
    'checkpoint_dir': 'checkpoints', # Directory that saves the checkpoints
    'sample_dir': 'samples'           # Directory to save the sample images
}
```

### Monitoring Training Progress

During training, check:
- Terminal output for loss values
- `samples_*/` directory for visual progress
- Training log file in `samples_*/training_log_*.txt`


### Checkpoint Files

Checkpoints are saved as `checkpoint_epoch_N.pth` and contain:
- Generator weights
- Discriminator weights
- Optimizer states
- Epoch number

## Running Inference

### Transform a Single Image

```bash
python inference.py --checkpoint checkpoints_5/checkpoint_epoch_100.pth \
                    --input my_photo.jpg \
                    --output cartoon_output.jpg
```

### Batch Process Multiple Images

Process all images in a directory:

```bash
python inference.py --checkpoint checkpoints_5/checkpoint_epoch_100.pth \
                    --input input_folder/ \
                    --output output_folder/
```

### Using Different Checkpoints

Compare results from different training stages:

```bash
# Early training stage
python inference.py --checkpoint checkpoints_5/checkpoint_epoch_10.pth \
                    --input photo.jpg --output result_early.jpg

# Mid training stage
python inference.py --checkpoint checkpoints_5/checkpoint_epoch_50.pth \
                    --input photo.jpg --output result_mid.jpg

# Final model
python inference.py --checkpoint checkpoints_5/checkpoint_epoch_100.pth \
                    --input photo.jpg --output result_final.jpg
```

### Available Checkpoint Directories

The project may contain multiple checkpoint directories:
- `checkpoints/` - Original training runs
- `checkpoints_2/`, `checkpoints_3/`, etc. - Additional experiments
- Each directory contains checkpoints from a different training session
- Made it so the Github Repo doesn't have one but should be in our shared work repo on the server

## Using the Web Interface

### Required Folder Structure for Web App

The web interface requires specific checkpoint folders to be present:

```
CSSE463-Non-photorealistic-Rendering/
├── The_Wind_Rises_Epoch/      #  checkpoints (required)
│   ├── checkpoint_epoch_X.pth
│   └── ...
├── Dragon_Ball_Epoch/         #  checkpoints (required)
│   ├── checkpoint_epoch_X.pth
    └── ...

```

**Download these folders from:**
https://drive.google.com/drive/folders/14Mo-VA6WtvTjSsU9crXexvtMdvJoD3VA?usp=sharing

Extract all 2 folders into your project root before starting the web server. This only has 24, 26, 28, 30

### Starting the Web Server

```bash
python app.py
```

The server will start on `http://localhost:5123`

### Web Interface Usage

1. Open your browser to `http://localhost:5123` (or the IP shown in terminal)
2. Click "Choose Image" to upload a photo
3. Select a style from the dropdown:
   - **The_Wind_Rises_Epoch** 
   - **Dragon_Ball_Epoch** 
4. Select an epoch checkpoint (higher epochs = more trained)
5. Adjust saturation and brightness sliders if desired
6. Click "Generate Cartoon" to process the image
7. View the result and click "Download Result" to save

### Web Interface Features

- Upload images up to 16MB
- Three different artistic styles to choose from
- Select from different training epochs (10, 20, 30, etc.)
- Real-time saturation and brightness adjustment
- Side-by-side comparison of original and result
- Download processed images
- Automatic model caching for faster processing
- Mobile-responsive design



## Model Evaluation

### Calculate FID Score

Evaluate model quality using Fréchet Inception Distance:

```bash
python FID_scorer.py
```


### Visual Evaluation

Check the `samples/` directories to visually assess:
- Style consistency
- Color accuracy
- Detail preservation
- Artifact presence

## Architecture Options

### Original Generator from Gao's Paper

**Command:**
```bash
python train.py
```

**Architecture:**
```
Input (256x256x3)
  ↓ Encoder (downsampling)
  ↓ ResBlocks (bottleneck)
  ↓ Decoder (upsampling)
Output (256x256x3)
```

### U-Net Generator

**Command:**
```bash
python train.py --use-unet
```

**Characteristics:**
- Encoder-decoder with skip connections
- Didn't quite turn the images into cartoon so we didn't go too far with it but it's here for reference

**Architecture:**
```
Input (256x256x3)
  ↓ Encoder ──────→ Skip connections ──────→ Decoder
  ↓ Bottleneck (ResBlocks)                      ↓
Output (256x256x3)
```


## Project Structure

```
CSSE463-Non-photorealistic-Rendering/
├── train.py                    # Training script
├── inference.py                # Image transformation script
├── generator.py                # Original generator & discriminator
├── generator_unet.py           # U-Net generator architecture
├── vgg19_loss.py              # VGG19 perceptual loss implementation
├── app.py                      # Flask web application (port 5123)
├── FID_scorer.py              # Model evaluation script
├── saturation.py              # Saturation loss utilities
├── templates/
│   └── index.html             # Web interface template
├── static/
│   ├── style.css              # Web interface styling
│   └── script.js              # Web interface logic
├── The_Wind_Rises_Epoch/      # Pre-trained anime style checkpoints (for web app)
├── Dragon_Ball_Epoch/         # Pre-trained cartoon style checkpoints (for web app)
├── checkpoints/             # Training checkpoints (various experiments)
├── samples/                 # Training sample outputs
├── train_photo/               # Training data: real photos
├── DB/                        # Training data: DragonBall cartoon images
├── TWR/                       # Training data: The Wind Rises anime images
└── README.md                  # This file
```

## Quick Start Guide

### 1. Prepare Data

```bash
mkdir train_photo DB
# Add your images to these directories (configurable weights):
- **GAN Loss (MSE/LSGAN)**: Weight = 1.0 - Adversarial training for realistic outputs
- **VGG19 Perceptual Loss**: Weight = 1.5 - Preserves content structure using conv4_4 features (CRITICAL)
- **Contrastive Loss (InfoNCE)**: Weight = 0.5 - Pulls outputs toward target style distribution
- **Identity/L1 Loss**: Weight = 5.0 - Preserves pixel-level content from input
- **Saturation Loss**: Weight = 2.0 - Enhances color vibrancy (currently disabled/commented out)

Current active losses: GAN + VGG19 + Contrastive + Identity

**Mini-batch size**: 4 images per batch
python train.py
```

### 3. Run Inference

```bash
python inference.py --checkpoint checkpoints_7/checkpoint_epoch_100.pth \
                    --input test.jpg \
                    --output result.jpg
```

### 4. Try Web Interface

Train your own model or download the pre-trained models from:
https://drive.google.com/drive/folders/14Mo-VA6WtvTjSsU9crXexvtMdvJoD3VA?usp=sharing

Then start the web server:

```bash
python app.py
# Open browser to http://localhost:5123
```

## Additional Information

### Loss Functions

The model uses multiple loss functions:
- **Adversarial Loss**: LSGAN for stable training
- **VGG19 Perceptual Loss**: Preserves content structure using conv4_4 features (CRITICAL)
- **Contrastive Loss**: Separates style and content features (optional)
- **Identity Loss**: Preserves input image content
- **Saturation Loss**: Enhances color vibrancy (optional)



---


