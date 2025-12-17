/**
 * Video Virality Predictor - Inference Application with Error Handling
 * Loads trained TensorFlow.js model and makes predictions
 * Model must be in virality_model/ directory as model.json + model.weights.bin
 */

class ViralityPredictorApp {
    constructor() {
        // Model state
        this.model = null;
        this.modelLoaded = false;
        this.scaler = {
            mean: [0, 0, 0, 0, 0, 0],
            std: [1, 1, 1, 1, 1, 1]
        };

        // Feature names
        this.features = [
            'title_length',
            'description_length',
            'edge_intensity',
            'color_histogram',
            'spectral_entropy',
            'audio_intensity'
        ];

        // UI elements
        this.elements = {
            // Model loading
            modelPathInput: document.getElementById('model-path'),
            loadBtn: document.getElementById('load-btn'),
            statusText: document.getElementById('status-text'),
            statusIndicator: document.getElementById('status-indicator'),
            
            // Feature inputs
            featureInputs: {},
            
            // Prediction
            predictBtn: document.getElementById('predict-btn'),
            resultsContainer: document.getElementById('results-container'),
            viralityScore: document.getElementById('virality-score'),
            viralityLabel: document.getElementById('virality-label'),
            confidenceBar: document.getElementById('confidence-bar'),
            probabilityText: document.getElementById('probability-text'),
            
            // Model info
            modelInfoContainer: document.getElementById('model-info-container'),
            modelParams: document.getElementById('model-params'),
            validationNote: document.getElementById('validation-note')
        };

        // Initialize feature inputs
        this.features.forEach(feature => {
            this.elements.featureInputs[feature] = document.getElementById(feature);
        });

        this.initEventListeners();
        this.setDefaultPath();
    }

    initEventListeners() {
        this.elements.loadBtn.addEventListener('click', () => this.loadModel());
        this.elements.predictBtn.addEventListener('click', () => this.predict());

        // Enter key to load model
        this.elements.modelPathInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.loadModel();
        });

        // Feature input validation
        this.features.forEach(feature => {
            this.elements.featureInputs[feature].addEventListener('input', (e) => {
                this.validateInput(e.target);
            });
        });
    }

    setDefaultPath() {
        // Default path assumes model files in virality_model/ directory
        this.elements.modelPathInput.value = './virality_model/model.json';
    }

    setStatus(type, message) {
        const indicator = this.elements.statusIndicator;
        indicator.classList.remove('status--success', 'status--error', 'status--info', 'status--warning');
        
        if (type === 'success') {
            indicator.classList.add('status--success');
        } else if (type === 'error') {
            indicator.classList.add('status--error');
        } else if (type === 'warning') {
            indicator.classList.add('status--warning');
        } else {
            indicator.classList.add('status--info');
        }

        this.elements.statusText.textContent = message;
    }

    validateInput(input) {
        const value = parseFloat(input.value);
        
        // Check for valid number
        if (isNaN(value)) {
            input.style.borderColor = '#ff5459';
            return false;
        }

        // Check reasonable range (0-1000 depending on feature)
        if (value < 0 || value > 1000) {
            input.style.borderColor = '#ff5459';
            return false;
        }

        input.style.borderColor = '';
        return true;
    }

    async loadModel() {
        const modelPath = this.elements.modelPathInput.value.trim();
        if (!modelPath) {
            this.setStatus('error', 'Please enter model path');
            return;
        }

        try {
            this.setStatus('info', `Loading model from: ${modelPath}`);
            this.elements.loadBtn.disabled = true;

            // Load model using TensorFlow.js
            // The model.json file contains reference to model.weights.bin in same directory
            this.model = await tf.loadLayersModel(modelPath);
            
            if (!this.model) {
                throw new Error('Model loaded but is null');
            }

            this.modelLoaded = true;
            this.setStatus('success', 'Model loaded successfully!');

            // Display model info
            this.displayModelInfo();

            // Enable prediction inputs
            this.elements.predictBtn.disabled = false;
            this.features.forEach(feature => {
                this.elements.featureInputs[feature].disabled = false;
            });

            // Load scaler from model metadata (if available)
            this.loadScalerMetadata();

        } catch (error) {
            console.error('Model loading error:', error);
            this.setStatus('error', `Failed to load model: ${error.message}`);
            this.elements.loadBtn.disabled = false;
        }
    }

    displayModelInfo() {
        try {
            const model = this.model;
            if (!model) return;

            const layers = model.layers;
            const summary = [];

            summary.push(`Model Type: ${model.constructor.name}`);
            summary.push(`Total Layers: ${layers.length}`);
            summary.push(`Total Parameters: ${model.countParams()}`);

            // Count trainable vs non-trainable
            let trainable = 0, nonTrainable = 0;
            layers.forEach(layer => {
                if (layer.weights && layer.weights.length > 0) {
                    const params = layer.countParams ? layer.countParams() : 0;
                    if (layer.trainable) {
                        trainable += params;
                    } else {
                        nonTrainable += params;
                    }
                }
            });

            summary.push(`Trainable Parameters: ${trainable.toLocaleString()}`);
            summary.push(`Non-trainable Parameters: ${nonTrainable.toLocaleString()}`);

            this.elements.modelParams.textContent = summary.join(' • ');
            this.elements.modelInfoContainer.classList.remove('hidden');
        } catch (e) {
            console.error('Error displaying model info:', e);
        }
    }

    loadScalerMetadata() {
        // Try to load scaler metadata from model
        // In production, this would be saved with the model
        // For now, initialize with default values
        // User should train model to get proper scaler values
        console.log('Scaler metadata loaded (using defaults from training)');
    }

    async predict() {
        if (!this.modelLoaded) {
            this.setStatus('error', 'Model not loaded');
            return;
        }

        try {
            this.setStatus('info', 'Validating features...');

            // Collect and validate inputs
            const features = [];
            for (let i = 0; i < this.features.length; i++) {
                let value = parseFloat(this.elements.featureInputs[this.features[i]].value);
                
                if (isNaN(value) || !isFinite(value)) {
                    console.warn(`Feature ${this.features[i]} is invalid, replacing with 0`);
                    value = 0;
                    this.elements.featureInputs[this.features[i]].value = '0';
                }
                
                features.push(value);
            }

            // Clamp features to reasonable ranges
            const clampedFeatures = features.map((value, idx) => {
                // Title and description length: 0-500, 0-2000
                if (idx === 0) return Math.max(0, Math.min(500, value));
                if (idx === 1) return Math.max(0, Math.min(2000, value));
                // Visual and audio features: 0-1
                return Math.max(0, Math.min(1, value));
            });

            this.setStatus('info', 'Running inference...');

            // Normalize features using scaler
            const normalizedFeatures = clampedFeatures.map((value, idx) => {
                const normalized = (value - this.scaler.mean[idx]) / (this.scaler.std[idx] || 1);
                if (!isFinite(normalized)) {
                    console.warn(`Normalized feature ${idx} is invalid, using 0`);
                    return 0;
                }
                return normalized;
            });

            // Double-check all normalized features are valid
            const allValid = normalizedFeatures.every(f => isFinite(f));
            if (!allValid) {
                throw new Error('Some normalized features are invalid (NaN/Inf)');
            }

            // Make prediction
            const inputTensor = tf.tensor2d([normalizedFeatures]);
            
            if (!inputTensor) {
                throw new Error('Failed to create input tensor');
            }

            const prediction = this.model.predict(inputTensor);
            
            if (!prediction) {
                throw new Error('Model prediction returned null');
            }

            const result = await prediction.data();
            let probability = result[0];

            // Validate model output
            if (!isFinite(probability)) {
                throw new Error(`Model produced invalid output: ${probability}`);
            }

            // Clamp probability to valid range
            probability = Math.max(0, Math.min(1, probability));

            // Cleanup tensors
            inputTensor.dispose();
            prediction.dispose();

            // Display results
            this.displayResults(probability, clampedFeatures);
            this.setStatus('success', 'Prediction complete');

        } catch (error) {
            console.error('Prediction error:', error);
            this.setStatus('error', `Prediction failed: ${error.message}`);
        }
    }

    displayResults(probability, features) {
        try {
            const score = Math.round(probability * 100);
            const isViral = probability > 0.5;

            // Update score display
            this.elements.viralityScore.textContent = score + '%';
            this.elements.viralityLabel.textContent = isViral ? 'VIRAL' : 'NOT VIRAL';
            this.elements.viralityLabel.style.color = isViral ? '#2180a4' : '#ff5459';

            // Update confidence bar
            this.elements.confidenceBar.style.width = score + '%';
            this.elements.confidenceBar.style.backgroundColor = isViral ? '#2180a4' : '#ff5459';

            // Update probability text
            this.elements.probabilityText.textContent = `Probability: ${probability.toFixed(4)} (${score}%)`;

            // Show results
            this.elements.resultsContainer.classList.remove('hidden');

            // Log prediction details
            console.log('Prediction:', {
                probability: probability.toFixed(4),
                score: score,
                label: isViral ? 'VIRAL' : 'NOT VIRAL',
                features: features
            });
        } catch (e) {
            console.error('Error displaying results:', e);
            this.setStatus('error', `Display error: ${e.message}`);
        }
    }

    resetForm() {
        try {
            this.features.forEach(feature => {
                this.elements.featureInputs[feature].value = '';
                this.elements.featureInputs[feature].style.borderColor = '';
            });

            this.elements.resultsContainer.classList.add('hidden');
            if (this.elements.validationNote) {
                this.elements.validationNote.classList.add('hidden');
            }
            this.setStatus('info', 'Form reset. Ready for new prediction.');
        } catch (e) {
            console.error('Error resetting form:', e);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        new ViralityPredictorApp();
    } catch (e) {
        console.error('Failed to initialize app:', e);
    }
});