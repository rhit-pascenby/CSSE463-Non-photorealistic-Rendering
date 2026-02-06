import torch
import torch.nn as nn
import torch.nn.functional as F


#This is the RESBLOCK from the research paper
class ResBlock(nn.Module):
    
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
     
        out = out + residual
        out = self.lrelu(out)
        
        return out


class Generator(nn.Module):

    
    def __init__(self):
        super(Generator, self).__init__()
        
      
        # Conv_n(32)k(7)s(1), LN, lRelu: 256x256x3 -> 256x256x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(64)k(3)s(2), LN, lRelu: 256x256x32 -> 128x128x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 128x128x64 -> 128x128x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(128)k(3)s(2), LN, lRelu: 128x128x64 -> 64x64x128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        
        # Conv_n(256)k(3)s(1), LN, lRelu: 64x64x128 -> 64x64x256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(256, affine=True)
        
        # Bottleneck: 4 ResBlocks at 64x64x256
        self.resblock1 = ResBlock(256)
        self.resblock2 = ResBlock(256)
        self.resblock3 = ResBlock(256)
        self.resblock4 = ResBlock(256)
        
      
      #decoding part
        # Conv_n(128)k(3)s(1), LN, lRelu: 64x64x256 -> 64x64x128
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.InstanceNorm2d(128, affine=True)
        
        # Upsample_n(128): 64x64x128 -> 128x128x128
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv_n(128)k(3)s(1), LN, lRelu: 128x128x128 -> 128x128x128
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.InstanceNorm2d(128, affine=True)
        
        # Upsample_n(64): 128x128x128 -> 256x256x64
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm8 = nn.InstanceNorm2d(64, affine=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 256x256x64 -> 256x256x64
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm9 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(64)k(3)s(1), LN, lRelu: 256x256x64 -> 256x256x64
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm10 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(32)k(7)s(1), LN, lRelu: 256x256x64 -> 256x256x32
        self.conv11 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.norm11 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(3)k(1)s(1), Tanh: 256x256x32 -> 256x256x3
        self.conv12 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        
        # Activation functions
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lrelu(x)
        
   
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
     
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.lrelu(x)
        
        x = self.upsample1(x)
        
        x = self.conv7(x)
        x = self.norm7(x)
        x = self.lrelu(x)
        
        x = self.conv8(x)
        x = self.norm8(x)
        x = self.lrelu(x)
        x = self.upsample2(x)
        
        x = self.conv9(x)
        x = self.norm9(x)
        x = self.lrelu(x)
        
        x = self.conv10(x)
        x = self.norm10(x)
        x = self.lrelu(x)
        
        x = self.conv11(x)
        x = self.norm11(x)
        x = self.lrelu(x)
        
        x = self.conv12(x)
        x = self.tanh(x)
        
        return x


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Conv_n(32)k(3)s(1), LN, lRelu: 256x256x3 -> 256x256x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        
        # Conv_n(64)k(3)s(2), LN, lRelu: 256x256x32 -> 128x128x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        
        # Conv_n(128)k(3)s(2), LN, lRelu: 128x128x64 -> 64x64x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        
        # Conv_n(256)k(3)s(2), LN, lRelu: 64x64x128 -> 32x32x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        
        # Conv_n(1)k(3)s(1): 32x32x256 -> 32x32x1
        self.conv5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        
        x = self.conv5(x)

        
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
