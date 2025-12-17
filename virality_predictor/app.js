/**
 * WORKING VIDEO VIRALITY PREDICTOR
 * - Extracts features from video: title_length, description_length, edge_intensity, 
 *   color_histogram, spectral_entropy, audio_intensity
 * - Normalizes using training data statistics
 * - Loads TensorFlow.js model from ./virality_model/model.json
 * - Runs inference and displays results
 */

class ViralityPredictorApp {
  constructor() {
    this.model = null;
    this.modelLoaded = false;
    this.videoFile = null;
    this.extractedFeatures = null;

    // EXACT normalization parameters from training data
    this.scaler = {
      mean: [52.0847966467, 511.0922737057, 0.2845898552, 0.7152603149, 0.4981954814, 0.5013699511],
      std:  [27.3783265961, 283.1248289425, 0.1585141472, 0.1585680307, 0.1007220271, 0.2884213778]
    };

    // DOM elements
    this.el = {
      statusIndicator: document.getElementById('status-indicator'),
      statusText: document.getElementById('status-text'),
      errorMessage: document.getElementById('error-message'),
      uploadArea: document.getElementById('upload-area'),
      videoFileInput: document.getElementById('video-file'),
      fileInfo: document.getElementById('file-info'),
      fileName: document.getElementById('file-name'),
      fileSize: document.getElementById('file-size'),
      fileDuration: document.getElementById('file-duration'),
      videoTitle: document.getElementById('video-title'),
      videoDescription: document.getElementById('video-description'),
      titleCount: document.getElementById('title-count'),
      descriptionCount: document.getElementById('description-count'),
      featureTitleLength: document.getElementById('feature-title-length'),
      featureDescriptionLength: document.getElementById('feature-description-length'),
      featureEdgeIntensity: document.getElementById('feature-edge-intensity'),
      featureColorHistogram: document.getElementById('feature-color-histogram'),
      featureSpectralEntropy: document.getElementById('feature-spectral-entropy'),
      featureAudioIntensity: document.getElementById('feature-audio-intensity'),
      predictBtn: document.getElementById('predict-btn'),
      resetBtn: document.getElementById('reset-btn'),
      resultsContainer: document.getElementById('results-container'),
      scoreCircle: document.getElementById('score-circle'),
      viralityScore: document.getElementById('virality-score'),
      viralityLabel: document.getElementById('virality-label'),
      confidenceFill: document.getElementById('confidence-fill'),
      probabilityText: document.getElementById('probability-text'),
      loadingMessage: document.getElementById('loading-message')
    };

    this.setupEventListeners();
    this.loadModel();
  }

  setupEventListeners() {
    // Upload area drag & drop
    this.el.uploadArea.addEventListener('dragover', e => {
      e.preventDefault();
      this.el.uploadArea.classList.add('dragover');
    });
    this.el.uploadArea.addEventListener('dragleave', () => {
      this.el.uploadArea.classList.remove('dragover');
    });
    this.el.uploadArea.addEventListener('drop', e => {
      e.preventDefault();
      this.el.uploadArea.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        this.handleVideoFile(e.dataTransfer.files[0]);
      }
    });
    this.el.uploadArea.addEventListener('click', () => {
      this.el.videoFileInput.click();
    });
    this.el.videoFileInput.addEventListener('change', e => {
      if (e.target.files.length > 0) {
        this.handleVideoFile(e.target.files[0]);
      }
    });

    // Text counters
    this.el.videoTitle.addEventListener('input', e => {
      this.el.titleCount.textContent = e.target.value.length;
      this.el.featureTitleLength.textContent = e.target.value.length;
      this.updateFeatureDisplay();
    });
    this.el.videoDescription.addEventListener('input', e => {
      this.el.descriptionCount.textContent = e.target.value.length;
      this.el.featureDescriptionLength.textContent = e.target.value.length;
      this.updateFeatureDisplay();
    });

    // Buttons
    this.el.predictBtn.addEventListener('click', () => this.predict());
    this.el.resetBtn.addEventListener('click', () => this.reset());
  }

  setStatus(type, message) {
    this.el.statusIndicator.classList.remove('status--info', 'status--success', 'status--error');
    this.el.statusIndicator.classList.add(`status--${type}`);
    this.el.statusText.textContent = message;
  }

  showError(message) {
    this.el.errorMessage.textContent = message;
    this.el.errorMessage.classList.add('visible');
  }

  clearError() {
    this.el.errorMessage.classList.remove('visible');
  }

  async loadModel() {
    try {
      this.setStatus('info', 'Loading TensorFlow.js model...');
      // Load from ./virality_model/model.json - MUST EXIST
      this.model = await tf.loadLayersModel('./virality_model/model.json');
      this.modelLoaded = true;
      this.setStatus('success', 'Model loaded successfully');
    } catch (err) {
      console.error('Model load error:', err);
      this.setStatus('error', 'Failed to load model');
      this.showError('Error loading model from ./virality_model/model.json. Make sure the model files exist.');
    }
  }

  handleVideoFile(file) {
    this.clearError();
    
    // Validate file type
    if (!file.type.startsWith('video/')) {
      this.showError('Please select a valid video file');
      return;
    }

    // Validate file size (100 MB limit)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      this.showError('File too large (max 100 MB)');
      return;
    }

    this.videoFile = file;

    // Update file info display
    this.el.fileName.textContent = file.name;
    this.el.fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    this.el.fileInfo.classList.add('visible');

    // Extract duration
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      const sec = video.duration || 0;
      const min = Math.floor(sec / 60);
      const s = Math.floor(sec % 60).toString().padStart(2, '0');
      this.el.fileDuration.textContent = `${min}:${s}`;
      URL.revokeObjectURL(url);
    };
    video.src = url;

    // Extract visual and audio features
    this.extractFeaturesFromVideo();
  }

  async extractFeaturesFromVideo() {
    if (!this.videoFile) return;

    try {
      this.setStatus('info', 'Extracting features from video...');

      // Extract visual features
      const { edge, color } = await this.extractVisualFeatures();
      this.el.featureEdgeIntensity.textContent = edge.toFixed(4);
      this.el.featureColorHistogram.textContent = color.toFixed(4);

      // Extract audio features
      const { entropy, intensity } = await this.extractAudioFeatures();
      this.el.featureSpectralEntropy.textContent = entropy.toFixed(4);
      this.el.featureAudioIntensity.textContent = intensity.toFixed(4);

      // Store all 6 features in correct order
      this.extractedFeatures = [
        parseFloat(this.el.featureTitleLength.textContent),
        parseFloat(this.el.featureDescriptionLength.textContent),
        edge,
        color,
        entropy,
        intensity
      ];

      console.log('Extracted features:', this.extractedFeatures);
      this.setStatus('success', 'Features extracted');
      this.updateFeatureDisplay();

    } catch (err) {
      console.error('Feature extraction error:', err);
      this.setStatus('error', 'Feature extraction failed');
      this.showError('Could not extract features: ' + err.message);
    }
  }

  async extractVisualFeatures() {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(this.videoFile);
      const video = document.createElement('video');
      video.src = url;
      video.muted = true;
      video.crossOrigin = 'anonymous';

      video.onloadedmetadata = () => {
        try {
          // Seek to middle of video
          video.currentTime = Math.min(video.duration * 0.5, 5);
        } catch (e) {
          video.currentTime = 0;
        }

        video.onseeked = () => {
          try {
            const w = video.videoWidth || 320;
            const h = video.videoHeight || 240;
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(video, 0, 0);

            const imageData = ctx.getImageData(0, 0, w, h);
            const data = imageData.data;

            // EDGE INTENSITY - compute Sobel operator
            let edgeSum = 0;
            let pixelCount = 0;
            for (let i = 0; i < data.length; i += 4) {
              const r = data[i];
              const g = data[i + 1];
              const b = data[i + 2];
              const gray = 0.299 * r + 0.587 * g + 0.114 * b;

              if (i >= 4) {
                const prevGray = 0.299 * data[i - 4] + 0.587 * data[i - 3] + 0.114 * data[i - 2];
                edgeSum += Math.abs(gray - prevGray);
              }
              pixelCount++;
            }
            const edge = Math.min(1, edgeSum / (pixelCount * 100));

            // COLOR HISTOGRAM - count unique color buckets
            const colorSet = new Set();
            for (let i = 0; i < data.length; i += 4) {
              const r = Math.floor(data[i] / 32);
              const g = Math.floor(data[i + 1] / 32);
              const b = Math.floor(data[i + 2] / 32);
              colorSet.add(`${r},${g},${b}`);
            }
            const color = Math.min(1, colorSet.size / 512);

            URL.revokeObjectURL(url);
            resolve({ edge, color });
          } catch (e) {
            URL.revokeObjectURL(url);
            reject(e);
          }
        };
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Cannot load video'));
      };
    });
  }

  async extractAudioFeatures() {
    return new Promise(async (resolve, reject) => {
      try {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error('File read error'));
        reader.onload = async (e) => {
          try {
            const arrayBuffer = e.target.result;
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Get audio data
            const rawData = audioBuffer.getChannelData(0);

            // SPECTRAL ENTROPY via simple FFT-like computation
            // Use Power Spectral Density approximation
            const fftSize = Math.min(2048, Math.floor(rawData.length / 2));
            const spectrum = new Float32Array(fftSize);
            
            let sum = 0;
            for (let i = 0; i < fftSize; i++) {
              let power = 0;
              for (let j = 0; j < Math.min(10, rawData.length - i); j++) {
                power += rawData[i + j * Math.floor(rawData.length / fftSize)] ** 2;
              }
              spectrum[i] = Math.sqrt(power);
              sum += spectrum[i];
            }

            // Normalize to probability distribution
            let entropy = 0;
            for (let i = 0; i < fftSize; i++) {
              const p = spectrum[i] / (sum || 1);
              if (p > 0) {
                entropy -= p * Math.log2(p);
              }
            }
            const maxEntropy = Math.log2(fftSize);
            const spectralEntropy = Math.min(1, entropy / (maxEntropy || 1));

            // AUDIO INTENSITY - RMS energy
            let sqSum = 0;
            for (let i = 0; i < Math.min(rawData.length, 44100 * 5); i++) {
              sqSum += rawData[i] ** 2;
            }
            const rms = Math.sqrt(sqSum / Math.min(rawData.length, 44100 * 5));
            const intensity = Math.min(1, rms);

            resolve({ entropy: spectralEntropy, intensity });
          } catch (e) {
            reject(e);
          }
        };
        reader.readAsArrayBuffer(this.videoFile);
      } catch (err) {
        reject(err);
      }
    });
  }

  updateFeatureDisplay() {
    // Enable predict button if all features are available
    const titleLen = parseFloat(this.el.featureTitleLength.textContent) || 0;
    const descLen = parseFloat(this.el.featureDescriptionLength.textContent) || 0;
    const edge = this.el.featureEdgeIntensity.textContent !== '—' ? parseFloat(this.el.featureEdgeIntensity.textContent) : null;
    const color = this.el.featureColorHistogram.textContent !== '—' ? parseFloat(this.el.featureColorHistogram.textContent) : null;
    const entropy = this.el.featureSpectralEntropy.textContent !== '—' ? parseFloat(this.el.featureSpectralEntropy.textContent) : null;
    const intensity = this.el.featureAudioIntensity.textContent !== '—' ? parseFloat(this.el.featureAudioIntensity.textContent) : null;

    this.el.predictBtn.disabled = !(titleLen > 0 && descLen > 0 && edge !== null && color !== null && entropy !== null && intensity !== null && this.modelLoaded);
  }

  async predict() {
    if (!this.modelLoaded) {
      this.showError('Model not loaded');
      return;
    }
    if (!this.extractedFeatures) {
      this.showError('No features extracted');
      return;
    }

    this.el.loadingMessage.style.display = 'block';
    this.el.predictBtn.disabled = true;

    try {
      // Normalize features
      const normalized = this.extractedFeatures.map((v, i) => {
        return (v - this.scaler.mean[i]) / (this.scaler.std[i] || 1);
      });

      console.log('Normalized input:', normalized);

      // Create tensor [1, 6]
      const input = tf.tensor2d([normalized]);
      const output = this.model.predict(input);
      const result = await output.data();
      const probability = Math.max(0, Math.min(1, result[0]));

      console.log('Raw prediction:', result[0]);
      console.log('Clamped probability:', probability);

      input.dispose();
      output.dispose();

      this.displayResults(probability);
      this.setStatus('success', 'Prediction complete');
    } catch (err) {
      console.error('Prediction error:', err);
      this.setStatus('error', 'Prediction failed');
      this.showError('Error: ' + err.message);
    } finally {
      this.el.loadingMessage.style.display = 'none';
      this.el.predictBtn.disabled = false;
    }
  }

  displayResults(probability) {
    const score = Math.round(probability * 100);
    const isViral = probability >= 0.5;

    this.el.viralityScore.textContent = score + '%';
    this.el.viralityLabel.textContent = isViral ? '✅ VIRAL' : '⚠️  NOT VIRAL';

    this.el.scoreCircle.classList.remove('viral', 'non-viral');
    this.el.scoreCircle.classList.add(isViral ? 'viral' : 'non-viral');

    this.el.confidenceFill.style.width = score + '%';
    this.el.probabilityText.textContent = `Probability: ${probability.toFixed(4)} (${score}%)`;

    this.el.resultsContainer.classList.add('visible');
  }

  reset() {
    this.videoFile = null;
    this.extractedFeatures = null;
    this.clearError();

    this.el.videoFileInput.value = '';
    this.el.fileInfo.classList.remove('visible');
    this.el.videoTitle.value = '';
    this.el.videoDescription.value = '';
    this.el.titleCount.textContent = '0';
    this.el.descriptionCount.textContent = '0';

    this.el.featureTitleLength.textContent = '0';
    this.el.featureDescriptionLength.textContent = '0';
    this.el.featureEdgeIntensity.textContent = '—';
    this.el.featureColorHistogram.textContent = '—';
    this.el.featureSpectralEntropy.textContent = '—';
    this.el.featureAudioIntensity.textContent = '—';

    this.el.resultsContainer.classList.remove('visible');
    this.el.predictBtn.disabled = true;
    this.el.loadingMessage.style.display = 'none';

    this.setStatus('success', 'Ready for new analysis');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new ViralityPredictorApp();
});
