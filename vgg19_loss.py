import torch
import torch.nn as nn
from torchvision import models


class VGG19PerceptualLoss(nn.Module):
    """
    VGG19 Perceptual Loss for content preservation.
    Extracts features from VGG19 conv4_4 layer (before activation).
    Based on the CTSS implementation.
    """
    
    def __init__(self):
        super(VGG19PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg19 = models.vgg19(pretrained=True)
        
        # Extract features up to conv4_4 (layer index 26 in features)
        # VGG19 layers: conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool2,
        #               conv3_1, conv3_2, conv3_3, conv3_4, pool3,
        #               conv4_1, conv4_2, conv4_3, conv4_4 (index 26)
        self.features = nn.Sequential(*list(vgg19.features.children())[:27])
        
        # Freeze VGG19 parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode
        self.features.eval()
        
        # VGG normalization values
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_vgg_input(self, x):
        """
        Normalize input from [-1, 1] to VGG19 input range.
        VGG expects images normalized with ImageNet mean/std.
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        
        # Normalize with ImageNet mean and std
        x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, real, generated):
        """
        Compute perceptual loss between real and generated images.
        
        Args:
            real: Real images in range [-1, 1], shape (B, 3, H, W)
            generated: Generated images in range [-1, 1], shape (B, 3, H, W)
        
        Returns:
            loss: L1 loss between VGG features
        """
        # Normalize inputs for VGG
        real_normalized = self.normalize_vgg_input(real)
        generated_normalized = self.normalize_vgg_input(generated)
        
        # Extract features from conv4_4
        real_features = self.features(real_normalized)
        generated_features = self.features(generated_normalized)
        
        # Compute L1 loss (as used in CTSS)
        loss = torch.mean(torch.abs(real_features - generated_features))
        
        return loss


def test_vgg_loss():
    """Test VGG19 perceptual loss."""
    print("Testing VGG19 Perceptual Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create loss module
    vgg_loss = VGG19PerceptualLoss().to(device)
    
    # Create sample images
    batch_size = 2
    real_img = torch.randn(batch_size, 3, 256, 256).to(device) * 2 - 1  # [-1, 1]
    fake_img = torch.randn(batch_size, 3, 256, 256).to(device) * 2 - 1  # [-1, 1]
    
    # Compute loss
    with torch.no_grad():
        loss = vgg_loss(real_img, fake_img)
    
    print(f"Input shape: {real_img.shape}")
    print(f"VGG Perceptual Loss: {loss.item():.6f}")
    
    # Test gradient flow
    fake_img.requires_grad = True
    loss = vgg_loss(real_img, fake_img)
    loss.backward()
    
    print(f"Gradient shape: {fake_img.grad.shape}")
    print(f"Gradient mean: {fake_img.grad.mean().item():.6f}")
    print("\nVGG19 Perceptual Loss test passed!")


if __name__ == "__main__":
    test_vgg_loss()
