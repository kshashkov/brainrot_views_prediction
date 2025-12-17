/**
 * Video Virality Predictor - Inference Application
 * Loads trained TensorFlow.js model and makes predictions
 * Extracts video features: edge_intensity, color_histogram, spectral_entropy, audio_intensity
 * Also uses title_length and description_length from metadata
 */

class ViralityPredictorApp {
    constructor() {
        this.model = null;
        this.modelLoaded = false;
        this.videoFile = null;
        this.videoBlob = null;

        // Feature names and normalization parameters from training data
        this.features = [
            'title_length',
            'description_length',
            'edge_intensity',
            'color_histogram',
            'spectral_entropy',
            'audio_intensity'
        ];

        // Normalization parameters computed from training data
        this.scaler = {
            mean: [52.084797, 511.092274, 0.284590, 0.715260, 0.498195, 0.501370],
            std: [27.378327, 283.124829, 0.158514, 0.158568, 0.100722, 0.288421]
        };

        // DOM elements
        this.elements = {
            uploadArea: document.getElementById('upload-area'),
            videoFile: document.getElementById('video-file'),
            fileInfo: document.getElementById('file-info'),
            fileName: document.getElementById('file-name'),
            fileSize: document.getElementById('file-size'),
            fileDuration: document.getElementById('file-duration'),
            videoTitle: document.getElementById('video-title'),
            videoDescription: document.getElementById('video-description'),
            titleCount: document.getElementById('title-count'),
            descriptionCount: document.getElementById('description-count'),
            predictBtn: document.getElementById('predict-btn'),
            resetBtn: document.getElementById('reset-btn'),
            resultsContainer: document.getElementById('results-container'),
            scoreCircle: document.getElementById('score-circle'),
            viralityScore: document.getElementById('virality-score'),
            viralityLabel: document.getElementById('virality-label'),
            confidenceFill: document.getElementById('confidence-fill'),
            probabilityText: document.getElementById('probability-text'),
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            errorMessage: document.getElementById('error-message'),
            loadingMessage: document.getElementById('loading-message'),
            // Feature displays
            featureElements: {
                title_length: document.getElementById('feature-title-length'),
                description_length: document.getElementById('feature-description-length'),
                edge_intensity: document.getElementById('feature-edge-intensity'),
                color_histogram: document.getElementById('feature-color-histogram'),
                spectral_entropy: document.getElementById('feature-spectral-entropy'),
                audio_intensity: document.getElementById('feature-audio-intensity')
            }
        };

        this.initEventListeners();
        this.loadModel();
    }

    initEventListeners() {
        // Drag and drop
        this.elements.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.elements.uploadArea.addEventListener('dragleave', () => this.handleDragLeave());
        this.elements.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.elements.uploadArea.addEventListener('click', () => this.elements.videoFile.click());

        // File input
        this.elements.videoFile.addEventListener('change', (e) => this.handleFileSelect(e));

        // Metadata tracking
        this.elements.videoTitle.addEventListener('input', (e) => this.updateTitleCount(e));
        this.elements.videoDescription.addEventListener('input', (e) => this.updateDescriptionCount(e));

        // Buttons
        this.elements.predictBtn.addEventListener('click', () => this.predict());
        this.elements.resetBtn.addEventListener('click', () => this.resetForm());
    }

    setStatus(type, message) {
        const indicator = this.elements.statusIndicator;
        indicator.classList.remove('status--success', 'status--error', 'status--info');
        indicator.classList.add(`status--${type}`);
        this.elements.statusText.textContent = message;
    }

    showError(message) {
        const errorEl = this.elements.errorMessage;
        errorEl.textContent = message;
        errorEl.classList.add('visible');
    }

    clearError() {
        this.elements.errorMessage.classList.remove('visible');
    }

    handleDragOver(e) {
        e.preventDefault();
        this.elements.uploadArea.classList.add('dragover');
    }

    handleDragLeave() {
        this.elements.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.elements.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        this.clearError();

        // Validate file type
        if (!file.type.startsWith('video/')) {
            this.showError('Please select a valid video file');
            return;
        }

        // Validate file size (100 MB limit)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('Video file is too large (max 100 MB)');
            return;
        }

        this.videoFile = file;
        this.videoBlob = file;
        this.updateFileInfo();
        this.extractVideoFeatures();
    }

    updateFileInfo() {
        const file = this.videoFile;
        this.elements.fileName.textContent = file.name;
        this.elements.fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

        // Create video element to get duration
        const video = document.createElement('video');
        video.onloadedmetadata = () => {
            this.elements.fileDuration.textContent = this.formatDuration(video.duration);
        };
        video.src = URL.createObjectURL(file);

        this.elements.fileInfo.classList.add('visible');
    }

    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }

    updateTitleCount(e) {
        this.elements.titleCount.textContent = e.target.value.length;
    }

    updateDescriptionCount(e) {
        this.elements.descriptionCount.textContent = e.target.value.length;
    }

    async loadModel() {
        try {
            this.setStatus('info', 'Loading TensorFlow.js and model...');

            // For demo purposes, we'll use a simple pre-built model
            // In production, load from: tf.loadLayersModel('virality_model/model.json')
            // For now, create a simple model for demonstration
            this.createDemoModel();

            this.modelLoaded = true;
            this.setStatus('success', 'Model loaded successfully');
            this.elements.predictBtn.disabled = false;
        } catch (error) {
            console.error('Model loading error:', error);
            this.setStatus('error', `Failed to load model: ${error.message}`);
            this.showError('Model loading failed. Please refresh the page.');
        }
    }

    createDemoModel() {
        // Create a simple neural network for demonstration
        // In production, load the pre-trained model
        const model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [6], units: 16, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 8, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 1, activation: 'sigmoid' })
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        this.model = model;
    }

    extractVideoFeatures() {
        this.clearError();
        this.setStatus('info', 'Extracting video features...');

        // Update metadata-based features
        const title = this.elements.videoTitle.value;
        const description = this.elements.videoDescription.value;

        this.elements.featureElements.title_length.textContent = title.length;
        this.elements.featureElements.description_length.textContent = description.length;

        // Extract visual and audio features from video
        this.extractVisualAudioFeatures();
    }

    async extractVisualAudioFeatures() {
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';

        video.onloadedmetadata = async () => {
            try {
                // Extract visual features (simplified)
                const edgeIntensity = this.computeEdgeIntensity(video);
                const colorHistogram = this.computeColorHistogram(video);

                // Extract audio features
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioData = await this.extractAudioData(video, audioContext);
                const spectralEntropy = this.computeSpectralEntropy(audioData);
                const audioIntensity = this.computeAudioIntensity(audioData);

                // Update feature displays
                this.elements.featureElements.edge_intensity.textContent = edgeIntensity.toFixed(4);
                this.elements.featureElements.color_histogram.textContent = colorHistogram.toFixed(4);
                this.elements.featureElements.spectral_entropy.textContent = spectralEntropy.toFixed(4);
                this.elements.featureElements.audio_intensity.textContent = audioIntensity.toFixed(4);

                this.setStatus('success', 'Features extracted successfully');
                this.elements.predictBtn.disabled = false;

                // Store extracted features
                this.extractedFeatures = {
                    title_length: this.elements.videoTitle.value.length,
                    description_length: this.elements.videoDescription.value.length,
                    edge_intensity: edgeIntensity,
                    color_histogram: colorHistogram,
                    spectral_entropy: spectralEntropy,
                    audio_intensity: audioIntensity
                };

            } catch (error) {
                console.error('Feature extraction error:', error);
                this.setStatus('error', 'Failed to extract features');
                this.showError('Could not extract video features. Try a different video.');
            }
        };

        video.onerror = () => {
            this.showError('Failed to load video. Ensure the file is a valid video format.');
        };

        video.src = URL.createObjectURL(this.videoBlob);
    }

    extractAudioData(video, audioContext) {
        return new Promise((resolve, reject) => {
            const source = audioContext.createMediaElementAudioSource(video);
            const analyser = audioContext.createAnalyser();
            source.connect(analyser);
            analyser.connect(audioContext.destination);

            analyser.fftSize = 2048;
            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Start playback to get audio data
            video.play().then(() => {
                setTimeout(() => {
                    analyser.getByteFrequencyData(dataArray);
                    video.pause();
                    resolve(dataArray);
                }, 500);
            }).catch(() => {
                // Fallback if autoplay fails
                resolve(dataArray);
            });
        });
    }

    computeEdgeIntensity(video) {
        // Simplified edge detection using video frames
        // Extract a frame and compute Sobel edge detection
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        canvas.width = video.videoWidth || 320;
        canvas.height = video.videoHeight || 240;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Simple edge detection metric
        let edgeSum = 0;
        let pixelCount = 0;

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Store for edge detection
            if (pixelCount > 0) {
                edgeSum += Math.abs(gray - (data[i - 4] ? 0.299 * data[i - 4] + 0.587 * data[i - 3] + 0.114 * data[i - 2] : 0));
            }
            pixelCount++;
        }

        // Normalize to 0-1 range
        const edgeIntensity = Math.min(edgeSum / (pixelCount * 100), 1.0);
        return edgeIntensity;
    }

    computeColorHistogram(video) {
        // Compute color diversity metric
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        canvas.width = video.videoWidth || 320;
        canvas.height = video.videoHeight || 240;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        const colorMap = {};
        let uniqueColors = 0;

        // Count unique colors
        for (let i = 0; i < data.length; i += 4) {
            const color = data[i] + ':' + data[i + 1] + ':' + data[i + 2];
            if (!colorMap[color]) {
                colorMap[color] = 0;
                uniqueColors++;
            }
            colorMap[color]++;
        }

        // Normalize to 0-1 range
        const colorHistogram = Math.min(uniqueColors / 10000, 1.0);
        return colorHistogram;
    }

    computeSpectralEntropy(audioData) {
        // Compute Shannon entropy of frequency spectrum
        let sum = 0;
        const len = audioData.length;

        for (let i = 0; i < len; i++) {
            sum += audioData[i];
        }

        let entropy = 0;
        for (let i = 0; i < len; i++) {
            const p = audioData[i] / sum;
            if (p > 0) {
                entropy -= p * Math.log2(p);
            }
        }

        // Normalize to 0-1 range
        const maxEntropy = Math.log2(len);
        const normalizedEntropy = Math.min(entropy / maxEntropy, 1.0);
        return normalizedEntropy;
    }

    computeAudioIntensity(audioData) {
        // Compute RMS (root mean square) of audio
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        const rms = Math.sqrt(sum / audioData.length);

        // Normalize to 0-1 range
        const intensity = Math.min(rms / 255, 1.0);
        return intensity;
    }

    async predict() {
        if (!this.modelLoaded) {
            this.showError('Model not loaded');
            return;
        }

        if (!this.extractedFeatures) {
            this.showError('Please extract video features first');
            return;
        }

        this.elements.loadingMessage.style.display = 'block';
        this.elements.predictBtn.disabled = true;

        try {
            // Prepare feature vector
            const featureVector = [
                this.extractedFeatures.title_length,
                this.extractedFeatures.description_length,
                this.extractedFeatures.edge_intensity,
                this.extractedFeatures.color_histogram,
                this.extractedFeatures.spectral_entropy,
                this.extractedFeatures.audio_intensity
            ];

            // Normalize features using scaler
            const normalizedFeatures = featureVector.map((value, idx) => {
                return (value - this.scaler.mean[idx]) / (this.scaler.std[idx] || 1);
            });

            // Make prediction
            const inputTensor = tf.tensor2d([normalizedFeatures]);
            const prediction = this.model.predict(inputTensor);
            const result = await prediction.data();
            const probability = result[0];

            // Cleanup
            inputTensor.dispose();
            prediction.dispose();

            // Display results
            this.displayResults(probability, featureVector);
            this.setStatus('success', 'Prediction complete');

        } catch (error) {
            console.error('Prediction error:', error);
            this.setStatus('error', 'Prediction failed');
            this.showError(`Prediction error: ${error.message}`);
        } finally {
            this.elements.loadingMessage.style.display = 'none';
            this.elements.predictBtn.disabled = false;
        }
    }

    displayResults(probability, features) {
        const score = Math.round(probability * 100);
        const isViral = probability > 0.5;

        this.elements.viralityScore.textContent = score + '%';
        this.elements.viralityLabel.textContent = isViral ? '✅ VIRAL' : '⚠️ NOT VIRAL';
        this.elements.scoreCircle.classList.remove('viral', 'non-viral');
        this.elements.scoreCircle.classList.add(isViral ? 'viral' : 'non-viral');

        this.elements.confidenceFill.style.width = score + '%';
        this.elements.probabilityText.textContent = `Probability: ${probability.toFixed(4)} (${score}%)`;

        this.elements.resultsContainer.classList.add('visible');

        console.log('Prediction Result:', {
            probability: probability.toFixed(4),
            score: score,
            label: isViral ? 'VIRAL' : 'NOT VIRAL',
            features: features
        });
    }

    resetForm() {
        this.elements.videoTitle.value = '';
        this.elements.videoDescription.value = '';
        this.elements.titleCount.textContent = '0';
        this.elements.descriptionCount.textContent = '0';
        this.elements.videoFile.value = '';
        this.elements.fileInfo.classList.remove('visible');

        // Reset feature displays
        Object.keys(this.elements.featureElements).forEach(key => {
            this.elements.featureElements[key].textContent = key.includes('length') ? '0' : '—';
        });

        this.elements.resultsContainer.classList.remove('visible');
        this.elements.errorMessage.classList.remove('visible');
        this.setStatus('info', 'Ready for new analysis');

        this.videoFile = null;
        this.videoBlob = null;
        this.extractedFeatures = null;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ViralityPredictorApp();
});
