from flask import Flask, render_template, request, send_file, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import io
import base64
import cv2
import numpy as np
from generator import Generator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model cache
model_cache = {}

def get_available_checkpoints():
    """Get list of available checkpoint directories and their epochs."""
    checkpoints = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Only look for these specific checkpoint directories
    allowed_folders = ['The_Wind_Rises_Epoch', 'Dragon_Ball_Epoch', 'Both_Style_Epoch']
    
    for folder_name in allowed_folders:
        checkpoint_dir = os.path.join(base_dir, folder_name)
        if os.path.isdir(checkpoint_dir):
            # Get all .pth files in this directory
            epochs = []
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth'):
                    epochs.append(file)
            if epochs:
                checkpoints[folder_name] = sorted(epochs)
    
    return checkpoints

def load_model(checkpoint_folder, checkpoint_file):
    """Load generator model from checkpoint."""
    cache_key = f"{checkpoint_folder}/{checkpoint_file}"
    
    # Check if model is already cached
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Cache the model (limit cache size to prevent memory issues)
    if len(model_cache) > 5:
        # Remove oldest entry
        model_cache.pop(next(iter(model_cache)))
    
    model_cache[cache_key] = (generator, device)
    return generator, device

def apply_saturation(pil_image, saturation_scale=1.0, value_scale=1.0):
    """Apply saturation and value adjustment to PIL image."""
    # Convert PIL to numpy array (RGB)
    img_array = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation (S channel is index 1) and value (V channel is index 2) independently
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_scale
    hsv[:, :, 2] = hsv[:, :, 2] * value_scale
    
    # Clip values to valid range [0, 255]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to uint8
    hsv = hsv.astype(np.uint8)
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert BGR back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(result_rgb)

def process_image(image_file, checkpoint_folder, checkpoint_file, saturation_scale=1.0, value_scale=1.0):
    """Process uploaded image and return cartoonized version."""
    # Load model
    generator, device = load_model(checkpoint_folder, checkpoint_file)
    
    # Load original image and get its size
    img = Image.open(image_file).convert('RGB')
    original_size = img.size  # Save original dimensions (width, height)
    
    # Prepare image for model (resize to 256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate cartoon-style image
    with torch.no_grad():
        output = generator(img_tensor)
    
    # Denormalize output
    output = (output + 1) / 2.0
    output = output.squeeze(0).cpu()
    
    # Convert to PIL Image
    output_img = transforms.ToPILImage()(output)
    
    # Resize output back to original input size
    output_img = output_img.resize(original_size, Image.LANCZOS)
    
    # Apply saturation/value adjustment if requested
    if saturation_scale != 1.0 or value_scale != 1.0:
        output_img = apply_saturation(output_img, saturation_scale, value_scale)
    
    # Convert to base64 for web display
    buffered = io.BytesIO()
    output_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

@app.route('/')
def index():
    """Render main page."""
    checkpoints = get_available_checkpoints()
    return render_template('index.html', checkpoints=checkpoints)

@app.route('/api/checkpoints')
def api_checkpoints():
    """API endpoint to get available checkpoints."""
    checkpoints = get_available_checkpoints()
    return jsonify(checkpoints)

@app.route('/process', methods=['POST'])
def process():
    """Process uploaded image."""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get checkpoint selection
        checkpoint_folder = request.form.get('checkpoint_folder')
        checkpoint_file = request.form.get('checkpoint_file')
        saturation_scale = float(request.form.get('saturation_scale', 1.0))
        value_scale = float(request.form.get('value_scale', 1.0))
        
        if not checkpoint_folder or not checkpoint_file:
            return jsonify({'error': 'Please select a checkpoint and epoch'}), 400
        
        # Verify checkpoint exists
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': 'Selected checkpoint does not exist'}), 404
        
        # Process image
        result_img = process_image(file, checkpoint_folder, checkpoint_file, saturation_scale, value_scale)
        
        return jsonify({
            'success': True,
            'image': result_img
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Cartoon Style Transfer Web App...")
    print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    app.run(debug=True, host='0.0.0.0', port=5123)
