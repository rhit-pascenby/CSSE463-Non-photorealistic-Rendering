// DOM elements
const imageInput = document.getElementById('imageInput');
const fileName = document.getElementById('fileName');
const checkpointFolder = document.getElementById('checkpointFolder');
const checkpointFile = document.getElementById('checkpointFile');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const originalPreview = document.getElementById('originalPreview');
const resultPreview = document.getElementById('resultPreview');
const downloadBtn = document.getElementById('downloadBtn');
const saturationSlider = document.getElementById('saturationSlider');
const saturationValue = document.getElementById('saturationValue');
const valueSlider = document.getElementById('valueSlider');
const valueValue = document.getElementById('valueValue');

let selectedImage = null;
let resultImage = null;

// Handle saturation slider
saturationSlider.addEventListener('input', (e) => {
    saturationValue.textContent = parseFloat(e.target.value).toFixed(1);
});

// Handle value slider
valueSlider.addEventListener('input', (e) => {
    valueValue.textContent = parseFloat(e.target.value).toFixed(1);
});

// Handle image selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        selectedImage = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            originalPreview.innerHTML = `<img src="${e.target.result}" alt="Original">`;
        };
        reader.readAsDataURL(file);
        
        checkProcessButton();
    }
});

// Handle checkpoint folder selection
checkpointFolder.addEventListener('change', (e) => {
    const folder = e.target.value;
    checkpointFile.innerHTML = '<option value="">-- Select Epoch --</option>';
    
    if (folder && checkpoints[folder]) {
        checkpointFile.disabled = false;
        
        // Sort epochs
        const epochs = checkpoints[folder].sort((a, b) => {
            // Extract epoch number for sorting
            const numA = parseInt(a.match(/\d+/)?.[0] || 0);
            const numB = parseInt(b.match(/\d+/)?.[0] || 0);
            return numA - numB;
        });
        
        epochs.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            checkpointFile.appendChild(option);
        });
    } else {
        checkpointFile.disabled = true;
    }
    
    checkProcessButton();
});

// Handle epoch selection
checkpointFile.addEventListener('change', checkProcessButton);

// Check if process button should be enabled
function checkProcessButton() {
    const hasImage = selectedImage !== null;
    const hasCheckpoint = checkpointFolder.value !== '' && checkpointFile.value !== '';
    processBtn.disabled = !(hasImage && hasCheckpoint);
}

// Handle process button click
processBtn.addEventListener('click', async () => {
    if (!selectedImage || !checkpointFolder.value || !checkpointFile.value) {
        showError('Please select an image and checkpoint configuration');
        return;
    }
    
    hideError();
    loading.classList.remove('hidden');
    processBtn.disabled = true;
    downloadBtn.classList.add('hidden');
    resultPreview.innerHTML = '<p class="placeholder">Processing...</p>';
    
    try {
        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('checkpoint_folder', checkpointFolder.value);
        formData.append('checkpoint_file', checkpointFile.value);
        formData.append('saturation_scale', saturationSlider.value);
        formData.append('value_scale', valueSlider.value);
        
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Processing failed');
        }
        
        if (data.success) {
            resultImage = data.image;
            resultPreview.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Result">`;
            downloadBtn.classList.remove('hidden');
        } else {
            throw new Error('Processing failed');
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
        resultPreview.innerHTML = '<p class="placeholder">Error processing image</p>';
    } finally {
        loading.classList.add('hidden');
        processBtn.disabled = false;
        checkProcessButton();
    }
});

// Handle download button click
downloadBtn.addEventListener('click', () => {
    if (resultImage) {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${resultImage}`;
        link.download = `cartoon_${Date.now()}.png`;
        link.click();
    }
});

// Utility functions
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
    errorDiv.textContent = '';
}

// Initialize
checkProcessButton();
