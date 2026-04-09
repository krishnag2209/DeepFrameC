const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.querySelector('.upload-content');
const previewSection = document.getElementById('preview-section');
const videoPreview = document.getElementById('video-preview');
const fileNameDisplay = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('result-section');

let selectedFile = null;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('dragover');
}

function unhighlight(e) {
    dropZone.classList.remove('dragover');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    let dt = e.dataTransfer;
    let files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        
        if (!selectedFile.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }

        uploadContent.style.display = 'none';
        previewSection.style.display = 'flex';
        resultSection.style.display = 'none';
        
        videoPreview.src = URL.createObjectURL(selectedFile);
        fileNameDisplay.textContent = selectedFile.name;
    }
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    previewSection.style.display = 'none';
    loader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `Server responded with status ${response.status}`);
        }

        const data = await response.json();
        
        if(data.error) {
            throw new Error(data.error);
        }

        displayResults(data);

    } catch (error) {
        alert('Analysis failed: ' + error.message);
        loader.style.display = 'none';
        previewSection.style.display = 'flex';
    }
});

function displayResults(data) {
    loader.style.display = 'none';
    resultSection.style.display = 'block';

    const verdictTarget = document.getElementById('verdict-target');
    if (data.verdict === 'FAKE') {
        verdictTarget.className = 'verdict-banner verdict-fake';
        verdictTarget.innerHTML = `⚠️ DEEPFAKE DETECTED ⚠️`;
    } else {
        verdictTarget.className = 'verdict-banner verdict-real';
        verdictTarget.innerHTML = `✅ REAL VIDEO ✅`;
    }

    const fakeBar = document.getElementById('fake-bar');
    const realBar = document.getElementById('real-bar');
    
    // Reset widths to 0 before animating
    fakeBar.style.width = '0%';
    realBar.style.width = '0%';
    
    // Animate bars with a slight delay
    setTimeout(() => {
        fakeBar.style.width = `${(data.fake_prob * 100).toFixed(1)}%`;
        realBar.style.width = `${(data.real_prob * 100).toFixed(1)}%`;
    }, 100);

    document.getElementById('fake-text').textContent = `${(data.fake_prob * 100).toFixed(2)}%`;
    document.getElementById('real-text').textContent = `${(data.real_prob * 100).toFixed(2)}%`;
    document.getElementById('time-text').textContent = `${data.elapsed.toFixed(1)}s`;
}
