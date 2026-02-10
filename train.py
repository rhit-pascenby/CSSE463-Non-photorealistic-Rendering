import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from generator import Generator, Discriminator
import torch.nn.functional as F
import argparse
import glob
from datetime import datetime


class ImageToImageDataset(Dataset):
    
    def __init__(self, normal_dir, anime_dir, image_size=256):
        self.normal_dir = normal_dir
        self.anime_dir = anime_dir
        
        # Get all image files
        self.normal_images = sorted([f for f in os.listdir(normal_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.anime_images = sorted([f for f in os.listdir(anime_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        

        self.length = min(len(self.normal_images), len(self.anime_images))
        
        print(f"Found {len(self.normal_images)} normal images and {len(self.anime_images)} anime images")
        print(f"Using {self.length} non-paired images for training")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # This is to Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
    
        normal_path = os.path.join(self.normal_dir, self.normal_images[idx])
        normal_img = Image.open(normal_path).convert('RGB')
        normal_img = self.transform(normal_img)
        
      
        anime_path = os.path.join(self.anime_dir, self.anime_images[idx])
        anime_img = Image.open(anime_path).convert('RGB')
        anime_img = self.transform(anime_img)
        
        return normal_img, anime_img


def rgb_to_hsv(rgb):
    """
    Convert RGB image tensor to HSV.
    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [-1, 1]
    Returns:
        hsv: Tensor of shape (B, 3, H, W) where H is [0, 1], S is [0, 1], V is [0, 1]
    """
    # Convert from [-1, 1] to [0, 1]
    rgb = (rgb + 1.0) / 2.0
    
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    
    max_rgb, argmax_rgb = rgb.max(1)
    min_rgb, _ = rgb.min(1)
    
    diff = max_rgb - min_rgb + 1e-7
    
    # Saturation
    s = diff / (max_rgb + 1e-7)
    
    # Value
    v = max_rgb
    
    # Hue
    h = torch.zeros_like(s)
    
    # Red is max
    mask = (argmax_rgb == 0)
    h[mask] = (((g - b) / diff)[mask] % 6) / 6.0
    
    # Green is max
    mask = (argmax_rgb == 1)
    h[mask] = (((b - r) / diff)[mask] + 2) / 6.0
    
    # Blue is max
    mask = (argmax_rgb == 2)
    h[mask] = (((r - g) / diff)[mask] + 4) / 6.0
    
    h = h % 1.0
    
    return torch.stack([h, s, v], dim=1)


class SaturationLoss(nn.Module):
    """Loss that encourages high saturation in generated images."""
    
    def __init__(self, target_saturation=0.7):
        super(SaturationLoss, self).__init__()
        self.target_saturation = target_saturation
    
    def forward(self, rgb_images):
        """
        Args:
            rgb_images: Generated images in RGB format, shape (B, 3, H, W), range [-1, 1]
        Returns:
            loss: Scalar loss that penalizes low saturation
        """
        # Convert to HSV
        hsv = rgb_to_hsv(rgb_images)
        saturation = hsv[:, 1, :, :]  # Extract saturation channel
        
        # Calculate mean saturation
        mean_saturation = saturation.mean()
        
        # Penalize if saturation is below target
        # Reward if saturation is above target (negative loss component)
        loss = F.relu(self.target_saturation - mean_saturation)
        
        # Also add a component that directly encourages higher saturation
        # Negative term means we want to maximize saturation
        saturation_boost = -saturation.mean() * 0.1
        
        return loss + saturation_boost


class ContrastiveLoss(nn.Module):
    """InfoNCE Contrastive Loss for image-to-image translation."""
    
    def __init__(self, temperature=0.07, feature_dim=256):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim
        
        # Feature projection head
        self.projection = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, anchor, positive, negatives=None):
        """
        Args:
            anchor: Generated images (fake_anime)
            positive: Target anime images
            negatives: Optional negative samples (if None, uses other samples in batch)
        """
        batch_size = anchor.size(0)
        
        # Extract features
        anchor_features = F.normalize(self.projection(anchor), dim=1)
        positive_features = F.normalize(self.projection(positive), dim=1)
        
        # Compute similarity scores
        # Positive pairs: anchor with its corresponding positive
        pos_sim = torch.sum(anchor_features * positive_features, dim=1) / self.temperature
        
        # Negative pairs: anchor with all other positives in the batch
        # Shape: [batch_size, batch_size]
        neg_sim = torch.matmul(anchor_features, positive_features.T) / self.temperature
        
        # Remove diagonal (positive pairs)
        mask = torch.eye(batch_size, device=anchor.device).bool()
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class GANTrainer:
   #based off research paper Gao22k, modified with contrastive loss
    
    def __init__(self, normal_dir, anime_dir, config=None):
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Default configuration
        self.config = {
            'image_size': 256,
            'batch_size': 4,  # Reduced from 4 to avoid OOM
            'num_epochs': 100,
            'lr_g': 0.0002,
            'lr_d': 0.0002,
            'beta1': 0.5,
            'beta2': 0.999,
            'lambda_contrastive': 1.0,  # Weight for contrastive loss
            'lambda_identity': 5.0,  # Weight for identity/content preservation
            'lambda_saturation': 2.0,  # Weight for saturation loss (encourages vivid colors)
            'target_saturation': 0.7,  # Target saturation level (0.7 = 70% saturation)
            'temperature': 0.07,  # Temperature for contrastive loss
            'save_interval': 10,
            'checkpoint_dir': 'checkpoints_6',
            'sample_dir': 'samples_6'
        }
        if config:
            self.config.update(config)
        
        # Create directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['sample_dir'], exist_ok=True)
        
        # Prepare log file path (will initialize after models are created)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.config['sample_dir'], f'training_log_{timestamp}.txt')
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize weights
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)
        
        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), 
                                      lr=self.config['lr_g'],
                                      betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(),
                                      lr=self.config['lr_d'],
                                      betas=(self.config['beta1'], self.config['beta2']))
        
        # Loss functions
        self.criterion_gan = nn.MSELoss()  # LSGAN loss (more stable)
        self.criterion_contrastive = ContrastiveLoss(
            temperature=self.config['temperature'],
            feature_dim=256
        ).to(self.device)
        self.criterion_l1 = nn.L1Loss()  # For optional identity loss
        # self.criterion_saturation = SaturationLoss(
        #     target_saturation=self.config['target_saturation']
        # ).to(self.device)
        
        # Dataset and DataLoader
        self.dataset = ImageToImageDataset(normal_dir, anime_dir, self.config['image_size'])
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.config['batch_size'],
                                     shuffle=True,
                                     num_workers=2)
        
        print(f"\nGenerator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Initialize log file after all components are ready
        self.init_log_file()
    
    @staticmethod
    def init_weights(m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def init_log_file(self):
        """Initialize log file with training configuration."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.config.items():
                f.write(f"{key:25s}: {value}\n")
            
            f.write("\nMODEL ARCHITECTURE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Generator parameters:     {sum(p.numel() for p in self.generator.parameters()):,}\n")
            f.write(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}\n")
            
            f.write("\nLOSS FUNCTIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("- GAN Loss: MSE (LSGAN)\n")
            f.write("- Contrastive Loss: InfoNCE\n")
            f.write("- Identity Loss: L1\n")
            f.write("- Saturation Loss: DISABLED (commented out)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("EPOCH RESULTS:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Epoch':>6} | {'D_loss':>8} | {'G_loss':>8} | {'GAN':>8} | {'Contrast':>8}\n")
            f.write("-" * 80 + "\n")
        
        print(f"Log file created: {self.log_file}")
    
    def log_epoch(self, epoch, avg_loss_d, avg_loss_g, avg_loss_gan, avg_loss_contrastive):
        """Log epoch metrics to file."""
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch+1:6d} | {avg_loss_d:8.4f} | {avg_loss_g:8.4f} | {avg_loss_gan:8.4f} | {avg_loss_contrastive:8.4f}\n")
    
    def train(self, start_epoch=0):
     
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Lambda Contrastive: {self.config['lambda_contrastive']}, Lambda Identity: {self.config['lambda_identity']}\n")  # , Lambda Saturation: {self.config['lambda_saturation']}
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.generator.train()
            self.discriminator.train()
            
            epoch_loss_g = 0
            epoch_loss_d = 0
            epoch_loss_contrastive = 0
            epoch_loss_gan = 0
            # epoch_loss_saturation = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for i, (normal_imgs, anime_imgs) in enumerate(pbar):
                normal_imgs = normal_imgs.to(self.device)
                anime_imgs = anime_imgs.to(self.device)
                
                batch_size = normal_imgs.size(0)
                
               
                real_label = torch.ones(batch_size, 1, 32, 32).to(self.device)
                fake_label = torch.zeros(batch_size, 1, 32, 32).to(self.device)
                
         
                self.optimizer_d.zero_grad()
                
                # Real anime images
                real_output = self.discriminator(anime_imgs)
                # Handle tuple return (output, attention_maps) if attention is enabled
                if isinstance(real_output, tuple):
                    real_output = real_output[0]
                loss_d_real = self.criterion_gan(real_output, real_label)
                
                # Fake anime images from generator
                fake_anime = self.generator(normal_imgs)
                fake_output = self.discriminator(fake_anime.detach())
                # Handle tuple return (output, attention_maps) if attention is enabled
                if isinstance(fake_output, tuple):
                    fake_output = fake_output[0]
                loss_d_fake = self.criterion_gan(fake_output, fake_label)
                
                # Total discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                self.optimizer_d.step()
                
               
                self.optimizer_g.zero_grad()
                
              
                fake_anime = self.generator(normal_imgs)
                
               
                fake_output = self.discriminator(fake_anime)
                # Handle tuple return (output, attention_maps) if attention is enabled
                if isinstance(fake_output, tuple):
                    fake_output, attention_maps = fake_output
                    # Optionally could add attention-based loss here
                loss_gan = self.criterion_gan(fake_output, real_label)
                
                # Contrastive loss: pull generated images close to target anime style
                # and push away from other anime images in batch
                loss_contrastive = self.criterion_contrastive(fake_anime, anime_imgs)
                
                # Optional: light identity loss to preserve some input structure
                loss_identity = self.criterion_l1(fake_anime, normal_imgs)
                
                # Saturation loss: encourage vivid, saturated colors like in cartoons/anime
                # loss_saturation = self.criterion_saturation(fake_anime)
                
                # Total generator loss
                # GAN loss: fool discriminator
                # Contrastive loss: match anime style distribution
                # Identity loss: preserve content structure (optional, can be removed)
                # Saturation loss: push colors to be more vibrant/saturated
                loss_g = (loss_gan + 
                         self.config['lambda_contrastive'] * loss_contrastive + 
                         self.config['lambda_identity'] * loss_identity)
                         # self.config['lambda_saturation'] * loss_saturation)
        
        
                loss_g.backward()
                self.optimizer_g.step()
                
                      # Update metrics
                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()
                epoch_loss_contrastive += loss_contrastive.item()
                epoch_loss_gan += loss_gan.item()
                # epoch_loss_saturation += loss_saturation.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{loss_d.item():.4f}',
                    'G_loss': f'{loss_g.item():.4f}',
                    'Contrast': f'{loss_contrastive.item():.4f}'
                    # 'Sat': f'{loss_saturation.item():.4f}'
                })
            
            # Calculate average losses
            num_batches = len(self.dataloader)
            avg_loss_g = epoch_loss_g / num_batches
            avg_loss_d = epoch_loss_d / num_batches
            avg_loss_gan = epoch_loss_gan / num_batches
            avg_loss_contrastive = epoch_loss_contrastive / num_batches
            # avg_loss_saturation = epoch_loss_saturation / num_batches
            
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                  f"D_loss: {avg_loss_d:.4f}, G_loss: {avg_loss_g:.4f}, "
                  f"Contrast: {avg_loss_contrastive:.4f}")  # , Sat: {avg_loss_saturation:.4f}
            
            # Log to file
            self.log_epoch(epoch, avg_loss_d, avg_loss_g, avg_loss_gan, avg_loss_contrastive)
            
            # Save samples and checkpoints
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_samples(epoch + 1, normal_imgs, anime_imgs, fake_anime)
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining completed!")
        self.save_checkpoint('final')
        
        # Final log entry
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def save_samples(self, epoch, normal_imgs, real_anime, fake_anime):
        """Save sample images during training."""
        import torchvision.utils as vutils
        
        self.generator.eval()
        with torch.no_grad():
            
            normal_imgs = (normal_imgs + 1) / 2.0
            real_anime = (real_anime + 1) / 2.0
            fake_anime = (fake_anime + 1) / 2.0
            
            # Save comparison
            comparison = torch.cat([normal_imgs[:4], fake_anime[:4], real_anime[:4]], dim=0)
            vutils.save_image(comparison,
                            os.path.join(self.config['sample_dir'], f'epoch_{epoch}.png'),
                            nrow=4,
                            normalize=False)
        self.generator.train()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoints."""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoints."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get('epoch', 0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train NPR GAN with optional checkpoint resumption')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint name or path to resume from (e.g., "checkpoint_epoch_50.pth" or "50" or full path)')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'image_size': 256,
        'batch_size': 4,          # Adjust based on GPU memory
        'num_epochs': 100,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_contrastive': 1.0,  # Weight for contrastive loss
        'lambda_identity': 5.0,      # Weight for identity preservation (can reduce or remove)
        'lambda_saturation': 2.0,    # Weight for saturation loss (higher = more saturated)
        'target_saturation': 0.7,    # Target saturation (0.7 = vivid cartoon colors)
        'temperature': 0.07,         # Temperature for contrastive loss
        'save_interval': 2,         # Save every N epochs
        'checkpoint_dir': 'checkpoints_6',
        'sample_dir': 'samples_6'
    }
    
    # Paths to your data
    normal_dir = 'train_photo'     # Folder with normal photos
    anime_dir = 'TWR'       # Folder with anime photos
    
    # Check if directories exist
    if not os.path.exists(normal_dir):
        print(f"Error: Normal photos directory '{normal_dir}' not found!")
        print("Please create the directory and add your normal photos.")
        return
    
    if not os.path.exists(anime_dir):
        print(f"Error: Anime photos directory '{anime_dir}' not found!")
        print("Please create the directory and add your anime photos.")
        return
    
    # Initialize trainer
    trainer = GANTrainer(normal_dir, anime_dir, config)
    
    # Handle checkpoint loading
    start_epoch = 0
    if args.checkpoint:
        checkpoint_path = None
        
        # If it's just a number, look for that epoch
        if args.checkpoint.isdigit():
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{args.checkpoint}.pth')
        # If it's a filename without path, look in checkpoint_dir
        elif not os.path.dirname(args.checkpoint):
            checkpoint_path = os.path.join(config['checkpoint_dir'], args.checkpoint)
        # Otherwise treat as full path
        else:
            checkpoint_path = args.checkpoint
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            print(f"Resuming training from epoch {start_epoch}\n")
        else:
            print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
            print("Starting training from scratch.\n")
    
    # Start training
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
