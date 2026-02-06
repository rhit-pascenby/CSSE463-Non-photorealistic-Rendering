import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
from generator import Generator


def load_model(checkpoint_path, device):
    """Load trained generator model."""
    generator = Generator().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return generator


def transform_image(image_path, generator, device, output_path=None):
    """Transform a single image using the trained generator."""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate anime-style image
    with torch.no_grad():
        output = generator(img_tensor)
    
    # Denormalize output
    output = (output + 1) / 2.0
    output = output.squeeze(0).cpu()
    
    # Convert to PIL Image
    output_img = transforms.ToPILImage()(output)
    
 
    if output_path:
        output_img.save(output_path)
        print(f"Saved: {output_path}")
    
    return output_img


def batch_transform(input_dir, output_dir, checkpoint_path):
    """Transform all images in a directory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    generator = load_model(checkpoint_path, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    print(f"\nTransforming {len(image_files)} images...")
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"anime_{img_file}")
        
        try:
            transform_image(input_path, generator, device, output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate anime-style images using trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image or directory')
    
    args = parser.parse_args()
    
    
    if os.path.isdir(args.input):
        batch_transform(args.input, args.output, args.checkpoint)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        generator = load_model(args.checkpoint, device)
        transform_image(args.input, generator, device, args.output)


if __name__ == "__main__":
    main()
