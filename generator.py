import torch
import torch.nn as nn
import torch.nn.functional as F


#This is the RESBLOCK from the research paper
class ResBlock(nn.Module):
    
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


class Generator(nn.Module):

    
    def __init__(self):
        super(Generator, self).__init__()
        
      
        # Conv_n(32)k(7)s(1), LN, lRelu: 256x256x3 -> 256x256x32
        # Use reflection padding to avoid edge artifacts
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(64)k(3)s(2), LN, lRelu: 256x256x32 -> 128x128x64
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 128x128x64 -> 128x128x64
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(128)k(3)s(2), LN, lRelu: 128x128x64 -> 64x64x128
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        
        # Conv_n(256)k(3)s(1), LN, lRelu: 64x64x128 -> 64x64x256
        self.pad5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.norm5 = nn.InstanceNorm2d(256, affine=True)
        
        # Bottleneck: 4 ResBlocks at 64x64x256
        self.resblock1 = ResBlock(256)
        self.resblock2 = ResBlock(256)
        self.resblock3 = ResBlock(256)
        self.resblock4 = ResBlock(256)
        
      
      #decoding part
        # Conv_n(128)k(3)s(1), LN, lRelu: 64x64x256 -> 64x64x128
        self.pad6 = nn.ReflectionPad2d(1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.norm6 = nn.InstanceNorm2d(128, affine=True)
        
        # Upsample_n(128): 64x64x128 -> 128x128x128
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv_n(128)k(3)s(1), LN, lRelu: 128x128x128 -> 128x128x128
        self.pad7 = nn.ReflectionPad2d(1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.norm7 = nn.InstanceNorm2d(128, affine=True)
        
        # Upsample_n(64): 128x128x128 -> 256x256x64
        self.pad8 = nn.ReflectionPad2d(1)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.norm8 = nn.InstanceNorm2d(64, affine=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 256x256x64 -> 256x256x64
        self.pad9 = nn.ReflectionPad2d(1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.norm9 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 256x256x64 -> 256x256x64
        self.pad10 = nn.ReflectionPad2d(1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.norm10 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(32)k(7)s(1), LN, lRelu: 256x256x64 -> 256x256x32
        # Use reflection padding to avoid edge artifacts
        self.pad11 = nn.ReflectionPad2d(3)
        self.conv11 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=0)
        self.norm11 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(3)k(1)s(1), Tanh: 256x256x32 -> 256x256x3
        self.conv12 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        
        # Activation functions
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        
        x = self.pad5(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lrelu(x)
        
   
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
     
        x = self.pad6(x)
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.lrelu(x)
        
        x = self.upsample1(x)
        
        x = self.pad7(x)
        x = self.conv7(x)
        x = self.norm7(x)
        x = self.lrelu(x)
        
        x = self.pad8(x)
        x = self.conv8(x)
        x = self.norm8(x)
        x = self.lrelu(x)
        x = self.upsample2(x)
        
        x = self.pad9(x)
        x = self.conv9(x)
        x = self.norm9(x)
        x = self.lrelu(x)
        
        x = self.pad10(x)
        x = self.conv10(x)
        x = self.norm10(x)
        x = self.lrelu(x)
        
        x = self.pad11(x)
        x = self.conv11(x)
        x = self.norm11(x)
        x = self.lrelu(x)
        
        x = self.conv12(x)
        x = self.tanh(x)
        
        return x


class SaliencyAttentionModule(nn.Module):
    """Attention module to identify salient regions in the image."""
    
    def __init__(self, in_channels):
        super(SaliencyAttentionModule, self).__init__()
        
        # Spatial attention branch
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention branch (squeeze-excitation style)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Spatial attention: identifies important spatial locations
        spatial_attn = self.spatial_attention(x)
        
        # Channel attention: identifies important feature channels
        channel_attn = self.channel_attention(x)
        
        # Apply both attentions
        x_attended = x * spatial_attn * channel_attn
        
        return x_attended, spatial_attn


class Discriminator(nn.Module):
    """PatchGAN Discriminator with Saliency Attention for focusing on important regions."""
    
    def __init__(self, use_attention=True):
        super(Discriminator, self).__init__()
        self.use_attention = use_attention
        
        # Conv_n(32)k(3)s(1), LN, lRelu: 256x256x3 -> 256x256x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(64)k(3)s(2), LN, lRelu: 256x256x32 -> 128x128x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(128)k(3)s(2), LN, lRelu: 128x128x64 -> 64x64x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        
        # Saliency attention at mid-level features
        if self.use_attention:
            self.attention1 = SaliencyAttentionModule(128)
        
        # Conv_n(256)k(3)s(2), LN, lRelu: 64x64x128 -> 32x32x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        
        # Another attention at deeper features
        if self.use_attention:
            self.attention2 = SaliencyAttentionModule(256)
        
        # Conv_n(1)k(3)s(1): 32x32x256 -> 32x32x1
        self.conv5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass with optional attention maps showing salient regions.
        Returns:
            output: 32x32x1 patch predictions
            attention_maps: list of attention maps (if use_attention=True)
        """
        attention_maps = []
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        
        # Apply first attention to focus on salient mid-level features
        if self.use_attention:
            x, attn_map1 = self.attention1(x)
            attention_maps.append(attn_map1)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        
        # Apply second attention to focus on salient high-level features
        if self.use_attention:
            x, attn_map2 = self.attention2(x)
            attention_maps.append(attn_map2)
        
        x = self.conv5(x)
        
        if self.use_attention:
            return x, attention_maps
        else:
            return x


def test_generator():
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Test Generator
    print("Testing Generator...")
    generator = Generator().to(device)
    
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = generator(input_tensor)
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    assert output.shape == (batch_size, 3, 256, 256), f"Expected output shape (1, 3, 256, 256), got {output.shape}"
    print("  Generator test passed!\n")
    
    # Test Discriminator
    print("Testing Discriminator...")
    discriminator = Discriminator().to(device)
    
    with torch.no_grad():
        disc_output = discriminator(input_tensor)
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {disc_output.shape}")
    print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    assert disc_output.shape == (batch_size, 1, 32, 32), f"Expected output shape (1, 1, 32, 32), got {disc_output.shape}"
    print("Discriminator test passed!\n")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_generator()
