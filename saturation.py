import cv2
import numpy as np
import argparse
import os


def increase_saturation(image_path, output_path, saturation_scale=1.3):
    """
    Increase the saturation of an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        saturation_scale: Multiplication factor for saturation (1.0 = no change, 2.0 = double saturation)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Increase saturation (S channel is index 1)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_scale
    hsv[:, :, 2] = hsv[:, :, 2] * saturation_scale
    
    # Clip values to valid range [0, 255]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to uint8
    hsv = hsv.astype(np.uint8)
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"Saved saturated image to: {output_path}")
    
    return result


def batch_increase_saturation(input_dir, output_dir, saturation_scale=1.3):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Processing {len(image_files)} images with saturation scale {saturation_scale}...")
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"saturated_{img_file}")
        
        try:
            increase_saturation(input_path, output_path, saturation_scale)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Increase image saturation')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image or directory')
    parser.add_argument('--scale', type=float, default=1.5,
                       help='Saturation scale factor (default: 1.5). 1.0 = no change, 2.0 = double saturation')
    
    args = parser.parse_args()
    
    # Check if input is directory or file
    if os.path.isdir(args.input):
        batch_increase_saturation(args.input, args.output, args.scale)
    else:
        increase_saturation(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()
