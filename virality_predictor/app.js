// app.js – end‑to‑end working implementation

class ViralityPredictorApp {
  constructor() {
    // Model & scaler (computed from train_data.csv)
    this.model = null;
    this.modelLoaded = false;

    // Order must match training: title_length, description_length, edge_intensity, color_histogram, spectral_entropy, audio_intensity
    this.scaler = {
      mean: [52.084797, 511.092274, 0.284590, 0.715260, 0.498195, 0.501370],
      std:  [27.378327, 283.124829, 0.158514, 0.158568, 0.100722, 0.288421]
    };

    this.videoFile = null;
    this.extractedFeatures = null;

    // Cache DOM
    this.el = {
      statusIndicator: document.getElementById("status-indicator"),
      statusText: document.getElementById("status-text"),
      uploadArea: document.getElementById("upload-area"),
      videoFile: document.getElementById("video-file"),
      fileMeta: document.getElementById("file-meta"),
      fileName: document.getElementById("file-name"),
      fileSize: document.getElementById("file-size"),
      fileDuration: document.getElementById("file-duration"),
      videoTitle: document.getElementById("video-title"),
      videoDescription: document.getElementById("video-description"),
      titleCount: document.getElementById("title-count"),
      descriptionCount: document.getElementById("description-count"),
      extractBtn: document.getElementById("extract-btn"),
      predictBtn: document.getElementById("predict-btn"),
      resetBtn: document.getElementById("reset-btn"),
      featureTitleLength: document.getElementById("feature-title-length"),
      featureDescriptionLength: document.getElementById("feature-description-length"),
      featureEdgeIntensity: document.getElementById("feature-edge-intensity"),
      featureColorHistogram: document.getElementById("feature-color-histogram"),
      featureSpectralEntropy: document.getElementById("feature-spectral-entropy"),
      featureAudioIntensity: document.getElementById("feature-audio-intensity"),
      errorMessage: document.getElementById("error-message"),
      loadingMessage: document.getElementById("loading-message"),
      results: document.getElementById("results"),
      scoreCircle: document.getElementById("score-circle"),
      viralityScore: document.getElementById("virality-score"),
      viralityLabel: document.getElementById("virality-label"),
      confidenceFill: document.getElementById("confidence-fill"),
      probabilityText: document.getElementById("probability-text")
    };

    this.initEvents();
    this.loadModel();
  }

  /* ---------- UI helpers ---------- */

  setStatus(type, msg) {
    this.el.statusIndicator.classList.remove("status--info", "status--success", "status--error");
    this.el.statusIndicator.classList.add(`status--${type}`);
    this.el.statusText.textContent = msg;
  }

  showError(msg) {
    this.el.errorMessage.textContent = msg;
    this.el.errorMessage.classList.add("visible");
  }

  clearError() {
    this.el.errorMessage.classList.remove("visible");
    this.el.errorMessage.textContent = "";
  }

  setLoading(on, msg) {
    if (on) {
      this.el.loadingMessage.textContent = msg || "Working…";
      this.el.loadingMessage.classList.add("visible");
    } else {
      this.el.loadingMessage.classList.remove("visible");
    }
  }

  initEvents() {
    // Drag & drop
    this.el.uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.el.uploadArea.classList.add("dragover");
    });
    this.el.uploadArea.addEventListener("dragleave", () => {
      this.el.uploadArea.classList.remove("dragover");
    });
    this.el.uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      this.el.uploadArea.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) this.handleFile(file);
    });
    // Click to open file
    this.el.uploadArea.addEventListener("click", () => this.el.videoFile.click());
    this.el.videoFile.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) this.handleFile(file);
    });

    // Title & description counters
    this.el.videoTitle.addEventListener("input", (e) => {
      this.el.titleCount.textContent = e.target.value.length;
    });
    this.el.videoDescription.addEventListener("input", (e) => {
      this.el.descriptionCount.textContent = e.target.value.length;
    });

    // Buttons
    this.el.extractBtn.addEventListener("click", () => this.extractAllFeatures());
    this.el.predictBtn.addEventListener("click", () => this.predict());
    this.el.resetBtn.addEventListener("click", () => this.reset());
  }

  /* ---------- Model loading ---------- */

  async loadModel() {
    try {
      this.setStatus("info", "Loading model from ./virality_model/model.json …");
      // IMPORTANT: model must exist in ./virality_model/model.json relative to index.html
      this.model = await tf.loadLayersModel("./virality_model/model.json");
      this.modelLoaded = true;
      this.setStatus("success", "Model loaded");
      // Enable feature extraction (requires video & metadata)
      this.updateButtons();
    } catch (err) {
      console.error(err);
      this.setStatus("error", "Failed to load model");
      this.showError("Could not load model from ./virality_model/model.json");
    }
  }

  /* ---------- File & metadata ---------- */

  handleFile(file) {
    this.clearError();
    if (!file.type.startsWith("video/")) {
      this.showError("Please upload a video file (mp4, webm, ogg, …)");
      return;
    }
    const maxBytes = 100 * 1024 * 1024;
    if (file.size > maxBytes) {
      this.showError("File too large (max 100 MB)");
      return;
    }
    this.videoFile = file;
    this.el.fileName.textContent = file.name;
    this.el.fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    this.el.fileMeta.classList.add("visible");
    this.extractedFeatures = null; // reset
    this.updateButtons();

    // Get duration
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const sec = video.duration || 0;
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60)
        .toString()
        .padStart(2, "0");
      this.el.fileDuration.textContent = `${m}:${s}`;
      URL.revokeObjectURL(url);
    };
    video.src = url;
  }

  updateButtons() {
    const hasVideo = !!this.videoFile;
    const hasTitle = this.el.videoTitle.value.length > 0;
    const hasDesc = this.el.videoDescription.value.length > 0;

    // Extract features requires video + metadata + model
    this.el.extractBtn.disabled = !(hasVideo && hasTitle && hasDesc && this.modelLoaded);

    // Predict requires features extracted
    this.el.predictBtn.disabled = !this.extractedFeatures || !this.modelLoaded;
  }

  /* ---------- Feature extraction pipeline ---------- */

  async extractAllFeatures() {
    if (!this.videoFile) {
      this.showError("Upload a video first");
      return;
    }
    if (!this.modelLoaded) {
      this.showError("Model is not loaded yet");
      return;
    }
    this.clearError();
    this.setLoading(true, "Extracting features from video…");
    this.setStatus("info", "Extracting features…");
    this.el.extractBtn.disabled = true;
    this.el.predictBtn.disabled = true;

    try {
      const title = this.el.videoTitle.value;
      const description = this.el.videoDescription.value;
      const titleLen = title.length;
      const descLen = description.length;

      // Update text features immediately
      this.el.featureTitleLength.textContent = titleLen;
      this.el.featureDescriptionLength.textContent = descLen;

      const { edgeIntensity, colorHist } = await this.extractVisualFeatures(this.videoFile);
      const { spectralEntropy, audioIntensity } = await this.extractAudioFeatures(this.videoFile);

      // Update UI
      this.el.featureEdgeIntensity.textContent = edgeIntensity.toFixed(4);
      this.el.featureColorHistogram.textContent = colorHist.toFixed(4);
      this.el.featureSpectralEntropy.textContent = spectralEntropy.toFixed(4);
      this.el.featureAudioIntensity.textContent = audioIntensity.toFixed(4);

      // Store in correct order for model
      this.extractedFeatures = [
        titleLen,
        descLen,
        edgeIntensity,
        colorHist,
        spectralEntropy,
        audioIntensity
      ];

      this.setStatus("success", "Features extracted successfully");
    } catch (err) {
      console.error(err);
      this.setStatus("error", "Feature extraction failed");
      this.showError("Feature extraction failed: " + (err.message || String(err)));
      this.extractedFeatures = null;
    } finally {
      this.setLoading(false);
      this.updateButtons();
    }
  }

  extractVisualFeatures(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      video.crossOrigin = "anonymous";

      video.onloadedmetadata = () => {
        const w = video.videoWidth || 320;
        const h = video.videoHeight || 240;
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const targetW = 160;
        const targetH = Math.round((h / w) * targetW) || 120;
        canvas.width = targetW;
        canvas.height = targetH;

        // Seek to 20% duration for a representative frame
        const t = (video.duration || 1) * 0.2;
        video.currentTime = t;

        video.onseeked = () => {
          try {
            ctx.drawImage(video, 0, 0, targetW, targetH);
            const img = ctx.getImageData(0, 0, targetW, targetH);
            const data = img.data;

            // Edge intensity: simple horizontal gradient magnitude
            let gradSum = 0;
            let count = 0;

            // Color histogram: number of distinct RGB buckets
            const colorBuckets = new Set();

            const width = targetW;
            const height = targetH;

            for (let y = 0; y < height; y++) {
              for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];

                // color bucket by downsampling
                const br = r >> 3;
                const bg = g >> 3;
                const bb = b >> 3;
                colorBuckets.add((br << 10) | (bg << 5) | bb);

                // gradient vs pixel to the right
                if (x < width - 1) {
                  const idx2 = (y * width + (x + 1)) * 4;
                  const r2 = data[idx2];
                  const g2 = data[idx2 + 1];
                  const b2 = data[idx2 + 2];
                  const g1 = 0.299 * r + 0.587 * g + 0.114 * b;
                  const g2Gray = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
                  gradSum += Math.abs(g1 - g2Gray);
                  count++;
                }
              }
            }
            const avgGrad = count > 0 ? gradSum / count : 0;
            const edgeIntensity = Math.max(0, Math.min(avgGrad / 40, 1)); // scale roughly into [0,1]

            const maxBuckets = 1024;
            const colorHist = Math.max(0, Math.min(colorBuckets.size / maxBuckets, 1));

            URL.revokeObjectURL(url);
            resolve({ edgeIntensity, colorHist });
          } catch (e) {
            URL.revokeObjectURL(url);
            reject(e);
          }
        };
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Cannot read video for visual features"));
      };
    });
  }

  extractAudioFeatures(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Failed to read video file for audio"));
      reader.onload = async (e) => {
        try {
          const arrayBuffer = e.target.result;
          const offlineCtx = new (window.OfflineAudioContext ||
            window.webkitOfflineAudioContext)(1, 44100 * 5, 44100); // up to 5s mono

          const audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer.slice(0, 44100 * 5 * 4).buffer || arrayBuffer);

          const source = offlineCtx.createBufferSource();
          source.buffer = audioBuffer;

          const analyser = offlineCtx.createAnalyser();
          analyser.fftSize = 2048;
          const freqBins = analyser.frequencyBinCount;
          const freqData = new Float32Array(freqBins);

          source.connect(analyser);
          analyser.connect(offlineCtx.destination);

          source.start(0);
          offlineCtx.startRendering().then(() => {
            analyser.getFloatFrequencyData(freqData);

            // Convert to magnitude [0,1]
            const mag = new Float32Array(freqBins);
            let magSum = 0;
            for (let i = 0; i < freqBins; i++) {
              const v = Math.pow(10, freqData[i] / 20); // dB to linear
              const clamped = Math.max(0, v);
              mag[i] = clamped;
              magSum += clamped;
            }
            if (magSum === 0) {
              resolve({ spectralEntropy: 0, audioIntensity: 0 });
              return;
            }

            // Spectral entropy
            let entropy = 0;
            for (let i = 0; i < freqBins; i++) {
              const p = mag[i] / magSum;
              if (p > 0) entropy -= p * Math.log2(p);
            }
            const maxEntropy = Math.log2(freqBins);
            const spectralEntropy = Math.max(0, Math.min(entropy / maxEntropy, 1));

            // Audio intensity (RMS in [0,1])
            let sqSum = 0;
            for (let i = 0; i < mag.length; i++) {
              sqSum += mag[i] * mag[i];
            }
            const rms = Math.sqrt(sqSum / mag.length);
            const audioIntensity = Math.max(0, Math.min(rms, 1));

            resolve({ spectralEntropy, audioIntensity });
          });
        } catch (err) {
          reject(err);
        }
      };
      reader.readAsArrayBuffer(file);
    });
  }

  /* ---------- Prediction ---------- */

  async predict() {
    if (!this.modelLoaded) {
      this.showError("Model not loaded");
      return;
    }
    if (!this.extractedFeatures) {
      this.showError("Extract features first");
      return;
    }

    this.clearError();
    this.setLoading(true, "Running model inference…");
    this.setStatus("info", "Predicting virality…");
    this.el.predictBtn.disabled = true;

    try {
      const feats = this.extractedFeatures.slice(); // [6]
      // Normalize
      const norm = feats.map((v, i) => (v - this.scaler.mean[i]) / (this.scaler.std[i] || 1));
      const input = tf.tensor2d([norm]); // shape [1,6]
      const out = this.model.predict(input);
      const data = await out.data();
      input.dispose();
      out.dispose();

      const p = Math.max(0, Math.min(1, data[0] || 0));
      this.displayResults(p);
      this.setStatus("success", "Prediction complete");
    } catch (err) {
      console.error(err);
      this.setStatus("error", "Prediction failed");
      this.showError("Prediction failed: " + (err.message || String(err)));
    } finally {
      this.setLoading(false);
      this.updateButtons();
    }
  }

  displayResults(prob) {
    const score = Math.round(prob * 100);
    const isViral = prob >= 0.5;

    this.el.viralityScore.textContent = `${score}%`;
    this.el.viralityLabel.textContent = isViral ? "VIRAL" : "NOT VIRAL";
    this.el.scoreCircle.classList.remove("viral", "non-viral");
    this.el.scoreCircle.classList.add(isViral ? "viral" : "non-viral");

    this.el.confidenceFill.style.width = `${score}%`;
    this.el.probabilityText.textContent = `Probability: ${prob.toFixed(4)} (${score}%)`;

    this.el.results.classList.add("visible");
  }

  /* ---------- Reset ---------- */

  reset() {
    this.videoFile = null;
    this.extractedFeatures = null;
    this.clearError();
    this.setStatus("info", "Ready");

    this.el.videoFile.value = "";
    this.el.fileMeta.classList.remove("visible");
    this.el.fileName.textContent = "–";
    this.el.fileSize.textContent = "–";
    this.el.fileDuration.textContent = "–";

    this.el.videoTitle.value = "";
    this.el.videoDescription.value = "";
    this.el.titleCount.textContent = "0";
    this.el.descriptionCount.textContent = "0";

    this.el.featureTitleLength.textContent = "0";
    this.el.featureDescriptionLength.textContent = "0";
    this.el.featureEdgeIntensity.textContent = "–";
    this.el.featureColorHistogram.textContent = "–";
    this.el.featureSpectralEntropy.textContent = "–";
    this.el.featureAudioIntensity.textContent = "–";

    this.el.results.classList.remove("visible");
    this.el.confidenceFill.style.width = "0%";
    this.el.viralityScore.textContent = "0%";
    this.el.viralityLabel.textContent = "NOT VIRAL";
    this.el.scoreCircle.classList.remove("viral", "non-viral");
    this.el.scoreCircle.classList.add("non-viral");

    this.updateButtons();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ViralityPredictorApp();
});
