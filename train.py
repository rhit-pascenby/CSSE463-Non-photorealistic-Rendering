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
        print(f"Using {self.length} paired images for training")
        
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


class GANTrainer:
   #based off research paper Gao22k
    
    def __init__(self, normal_dir, anime_dir, config=None):
        self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
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
            'lambda_l1': 100.0,  # Weight for L1 loss (higher = more like target)
            'save_interval': 10,
            'checkpoint_dir': 'checkpoints',
            'sample_dir': 'samples_2'
        }
        if config:
            self.config.update(config)
        
        # Create directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['sample_dir'], exist_ok=True)
        
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
        self.criterion_l1 = nn.L1Loss()
        
        # Dataset and DataLoader
        self.dataset = ImageToImageDataset(normal_dir, anime_dir, self.config['image_size'])
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.config['batch_size'],
                                     shuffle=True,
                                     num_workers=2)
        
        print(f"\nGenerator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
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
    
    def train(self):
     
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Lambda L1: {self.config['lambda_l1']}\n")
        
        for epoch in range(self.config['num_epochs']):
            self.generator.train()
            self.discriminator.train()
            
            epoch_loss_g = 0
            epoch_loss_d = 0
            epoch_loss_l1 = 0
            epoch_loss_gan = 0
            
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
                loss_d_real = self.criterion_gan(real_output, real_label)
                
                # Fake anime images from generator
                fake_anime = self.generator(normal_imgs)
                fake_output = self.discriminator(fake_anime.detach())
                loss_d_fake = self.criterion_gan(fake_output, fake_label)
                
                # Total discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                self.optimizer_d.step()
                
               
                self.optimizer_g.zero_grad()
                
              
                fake_anime = self.generator(normal_imgs)
                
               
                fake_output = self.discriminator(fake_anime)
                loss_gan = self.criterion_gan(fake_output, real_label)
                
            
                loss_identity = self.criterion_l1(fake_anime, normal_imgs)
                loss_l1 = torch.tensor(0.0).to(self.device)  # For logging compatibility
                
                # Total generator loss
                # GAN loss: make it look anime
                # Identity loss: keep the content/structure of input
                loss_g = loss_gan + 10.0 * loss_identity
        
        
                loss_g.backward()
                self.optimizer_g.step()
                
                      # Update metrics
                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()
                epoch_loss_l1 += loss_l1.item()
                epoch_loss_gan += loss_gan.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{loss_d.item():.4f}',
                    'G_loss': f'{loss_g.item():.4f}',
                    'L1': f'{loss_l1.item():.4f}'
                })
            
            # Calculate average losses
            num_batches = len(self.dataloader)
            avg_loss_g = epoch_loss_g / num_batches
            avg_loss_d = epoch_loss_d / num_batches
            avg_loss_gan = epoch_loss_gan / num_batches
            
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                  f"D_loss: {avg_loss_d:.4f}, G_loss: {avg_loss_g:.4f}")
            
            # Save samples and checkpoints
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_samples(epoch + 1, normal_imgs, anime_imgs, fake_anime)
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining completed!")
        self.save_checkpoint('final')
    
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
    
    # Configuration
    config = {
        'image_size': 256,
        'batch_size': 4,          # Adjust based on GPU memory
        'num_epochs': 100,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_l1': 100.0,      
        'save_interval': 10,       # Save every N epochs
        'checkpoint_dir': 'checkpoints',
        'sample_dir': 'samples'
    }
    
    # Paths to your data
    normal_dir = 'train_photo'     # Folder with normal photos
    anime_dir = 'DB'       # Folder with anime photos
    
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
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
