import os
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from generator import Generator

def compute_fid(cartoon_dir, generated_dir, batch_size=32):
    if not os.path.isdir(cartoon_dir) or not os.path.isdir(generated_dir):
        raise FileNotFoundError("Both cartoon_dir and generated_dir must exist")

    def list_files(d):
        return sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])

    cartoon_files = list_files(cartoon_dir)
    generated_files = list_files(generated_dir)
    if len(cartoon_files) < 2 or len(generated_files) < 2:
        raise ValueError("Need at least 2 images in each directory to compute FID")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # convert to uint8 [0,255]
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    def update_from_paths(paths, real):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(transform(img))
                except Exception:
                    continue
            if not imgs:
                continue
            batch = torch.stack(imgs, dim=0).to(device)
            fid.update(batch, real=real)

    with torch.no_grad():
        update_from_paths(cartoon_files, real=True)
        update_from_paths(generated_files, real=False)

    return float(fid.compute())

def generate_images(checkpoint_path, input_dir, output_dir, num_images=None, img_size=256):
    """Generate images using checkpoint weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load generator
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"Loaded generator from {checkpoint_path}")
    
    # Get input files
    def list_files(d):
        return sorted([os.path.join(d, f) for f in os.listdir(d) 
                      if os.path.isfile(os.path.join(d, f))])
    
    input_files = list_files(input_dir)
    if num_images:
        input_files = input_files[:num_images]
    
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    print(f"Generating {len(input_files)} images...")
    with torch.no_grad():
        for i, fpath in enumerate(input_files):
            try:
                img = Image.open(fpath).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                out = generator(x)
                out = (out + 1) / 2.0  # denormalize to [0,1]
                out = (out * 255).to(torch.uint8)  # convert to uint8
                out_img = transforms.ToPILImage()(out.squeeze(0).cpu())
                fname = os.path.basename(fpath)
                out_img.save(os.path.join(output_dir, fname))
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{len(input_files)}")
            except Exception as e:
                print(f"  Error processing {fpath}: {e}")
    
    print(f"Generated {len(input_files)} images to {output_dir}")

if __name__ == "__main__":
    CARTOON_DIR = "TWR"
    GENERATED_DIR = "generated_FID"
    CHECKPOINT = "checkpoints_4/checkpoint_epoch_100.pth"
    INPUT_DIR = "train_photo"
    BATCH_SIZE = 32

    # Count images in TWR
    num_twr = len([f for f in os.listdir(CARTOON_DIR) if os.path.isfile(os.path.join(CARTOON_DIR, f))])
    print(f"Found {num_twr} images in {CARTOON_DIR}\n")
    
    # Generate same number from train_photo
    print(f"Generating {num_twr} images from {INPUT_DIR}...\n")
    generate_images(CHECKPOINT, INPUT_DIR, GENERATED_DIR, num_images=num_twr)
    
    # Compute FID
    print(f"\nComputing FID...")
    score = compute_fid(CARTOON_DIR, GENERATED_DIR, batch_size=BATCH_SIZE)
    print(f"FID score: {score:.4f}")