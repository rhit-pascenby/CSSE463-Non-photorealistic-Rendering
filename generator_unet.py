import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with reflection padding and instance normalization."""
    
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
    
    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.lrelu(out)
        
        return out


class DownBlock(nn.Module):
    """Encoder block with downsampling."""
    
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DownBlock, self).__init__()
        self.downsample = downsample
        
        if downsample:
            self.pad = nn.ReflectionPad2d(1)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.pad = nn.ReflectionPad2d(1)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class UpBlock(nn.Module):
    """Decoder block with upsampling and skip connections."""
    
    def __init__(self, in_channels, out_channels, upsample=True):
        super(UpBlock, self).__init__()
        self.upsample_layer = None
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # After concatenation with skip connection, channels will be in_channels + skip_channels
        # We'll handle this in the forward pass
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, skip=None):
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net style generator with skip connections for cartoon style transfer.
    
    Architecture:
        Encoder: Progressively downsamples and increases channels
        Bottleneck: ResBlocks for feature transformation
        Decoder: Progressively upsamples with skip connections from encoder
    """
    
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        # ==================== ENCODER (Contracting Path) ====================
        # Initial conv: 256x256x3 -> 256x256x32
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder block 1: 256x256x32 -> 128x128x64
        self.enc1 = DownBlock(32, 64, downsample=True)
        
        # Encoder block 2: 128x128x64 -> 128x128x64 (no downsample)
        self.enc2 = DownBlock(64, 64, downsample=False)
        
        # Encoder block 3: 128x128x64 -> 64x64x128
        self.enc3 = DownBlock(64, 128, downsample=True)
        
        # Encoder block 4: 64x64x128 -> 64x64x256
        self.enc4 = DownBlock(128, 256, downsample=False)
        
        # ==================== BOTTLENECK ====================
        # ResBlocks at 64x64x256
        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256)
        )
        
        # ==================== DECODER (Expanding Path) ====================
        # Decoder block 1: 64x64x256 -> 64x64x128 (with skip from enc4: 256+256=512 input)
        self.dec1 = UpBlock(512, 128, upsample=False)  # 512 because of skip connection
        
        # Decoder block 2: 64x64x128 -> 128x128x128 (with skip from enc2: 128+64=192 input)
        self.dec2 = UpBlock(192, 128, upsample=True)   # 192 because of skip connection (128 from dec1 + 64 from enc2)
        
        # Decoder block 3: 128x128x128 -> 128x128x64 (with skip from enc1: 128+64=192 input)
        self.dec3 = UpBlock(192, 64, upsample=False)   # 192 because of skip connection
        
        # Decoder block 4: 128x128x64 -> 256x256x64 (with skip from enc0: 64+32=96 input)
        self.dec4 = UpBlock(96, 64, upsample=True)     # 96 because of skip connection
        
        # Decoder block 5: 256x256x64 -> 256x256x32 (no skip connection)
        self.dec5 = UpBlock(64, 32, upsample=False)    # No skip connection for final layer
        
        # ==================== OUTPUT ====================
        # Final conv: 256x256x32 -> 256x256x3
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input image tensor (batch_size, 3, 256, 256)
            
        Returns:
            Output cartoon-style image (batch_size, 3, 256, 256)
        """
        # ==================== ENCODER ====================
        # Save intermediate features for skip connections
        enc0_out = self.enc0(x)        # 256x256x32
        enc1_out = self.enc1(enc0_out) # 128x128x64
        enc2_out = self.enc2(enc1_out) # 128x128x64
        enc3_out = self.enc3(enc2_out) # 64x64x128
        enc4_out = self.enc4(enc3_out) # 64x64x256
        
        # ==================== BOTTLENECK ====================
        bottleneck_out = self.bottleneck(enc4_out)  # 64x64x256
        
        # ==================== DECODER with SKIP CONNECTIONS ====================
        # Skip connection: concatenate with enc4_out (both 64x64)
        dec1_out = self.dec1(bottleneck_out, enc4_out)  # 64x64x128
        
        # Upsample to 128x128, then skip with enc2_out (both 128x128)
        dec2_out = self.dec2(dec1_out, enc2_out)        # 128x128x128
        
        # Skip connection: concatenate with enc1_out (both 128x128)
        dec3_out = self.dec3(dec2_out, enc1_out)        # 128x128x64
        
        # Upsample to 256x256, then skip with enc0_out (both 256x256)
        dec4_out = self.dec4(dec3_out, enc0_out)        # 256x256x64
        
        # No skip connection for final refinement layer
        dec5_out = self.dec5(dec4_out)                  # 256x256x32
        
        # ==================== OUTPUT ====================
        output = self.output(dec5_out)                   # 256x256x3
        
        return output


def test_unet_generator():
    """Test the U-Net generator architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Testing U-Net Generator...")
    generator = UNetGenerator().to(device)
    
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = generator(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    assert output.shape == (batch_size, 3, 256, 256), \
        f"Expected output shape ({batch_size}, 3, 256, 256), got {output.shape}"
    
    print("\n✓ U-Net Generator test passed!")
    print("\nKey U-Net Features:")
    print("  ✓ Encoder-Decoder architecture")
    print("  ✓ Skip connections between encoder and decoder")
    print("  ✓ Preserves fine-grained details from encoder")
    print("  ✓ Symmetric architecture")


if __name__ == "__main__":
    test_unet_generator()
