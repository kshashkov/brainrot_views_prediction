/**
 * Video Virality Predictor - Neural Network Training Engine
 * All processing happens client-side in the browser using TensorFlow.js
 */

class ViralityPredictor {
  constructor() {
    // Feature configuration
    this.features = [
      'title_length',
      'description_length',
      'edge_intensity',
      'color_histogram',
      'spectral_entropy',
      'audio_intensity'
    ];

    // Data storage
    this.data = null;
    this.trainX = null;
    this.trainY = null;
    this.valX = null;
    this.valY = null;
    this.scaler = {
      mean: null,
      std: null
    };

    // Model state
    this.model = null;
    this.isTraining = false;
    this.history = {
      loss: [],
      val_loss: [],
      acc: [],
      val_acc: [],
      auc: []
    };

    // UI elements
    this.elements = {
      dataInput: document.getElementById('data-input'),
      dataPreview: document.getElementById('data-preview'),
      epochs: document.getElementById('epochs'),
      batchSize: document.getElementById('batch-size'),
      learningRate: document.getElementById('learning-rate'),
      trainBtn: document.getElementById('train-btn'),
      stopBtn: document.getElementById('stop-btn'),
      statusIndicator: document.getElementById('status-indicator'),
      statusText: document.getElementById('status-text'),
      trainingProgress: document.getElementById('training-progress'),
      progressBar: document.getElementById('progress-bar'),
      epochDisplay: document.getElementById('epoch-display'),
      metricsPlaceholder: document.getElementById('metrics-placeholder'),
      metricsContainer: document.getElementById('metrics-container'),
      chartPlaceholder: document.getElementById('chart-placeholder'),
      chartContainer: document.getElementById('chart-container'),
      trainLoss: document.getElementById('train-loss'),
      valLoss: document.getElementById('val-loss'),
      trainAcc: document.getElementById('train-acc'),
      valAcc: document.getElementById('val-acc'),
      valAuc: document.getElementById('val-auc'),
      lossCanvas: document.getElementById('loss-chart'),
      accCanvas: document.getElementById('acc-chart'),
      exportBtn: document.getElementById('export-btn'),
      exportInfo: document.getElementById('export-info')
    };

    this.initEventListeners();
  }

  initEventListeners() {
    this.elements.dataInput.addEventListener('change', (e) => this.handleFileUpload(e));
    this.elements.trainBtn.addEventListener('click', () => this.startTraining());
    this.elements.stopBtn.addEventListener('click', () => this.stopTraining());
    this.elements.exportBtn.addEventListener('click', () => this.exportModel());
  }

  setStatus(type, message) {
    const indicator = this.elements.statusIndicator;
    indicator.classList.remove('status--success', 'status--error');
    if (type === 'success') indicator.classList.add('status--success');
    if (type === 'error') indicator.classList.add('status--error');
    this.elements.statusText.textContent = message;
  }

  async handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      this.setStatus('info', 'Loading data...');
      const text = await file.text();
      const rows = text.trim().split('\n');
      const headers = rows[0].split(',');

      // Parse CSV
      const data = [];
      for (let i = 1; i < rows.length; i++) {
        const values = rows[i].split(',').map(v => parseFloat(v.trim()));
        if (values.length === headers.length) {
          const row = {};
          headers.forEach((h, j) => {
            row[h.trim()] = values[j];
          });
          data.push(row);
        }
      }

      if (data.length === 0) {
        throw new Error('No valid data rows found');
      }

      // Verify all required features present
      const missingFeatures = this.features.filter(f => !data[0].hasOwnProperty(f));
      if (missingFeatures.length > 0) {
        throw new Error(`Missing features: ${missingFeatures.join(', ')}`);
      }

      if (!data[0].hasOwnProperty('virality')) {
        throw new Error('Missing target column: virality');
      }

      this.data = data;
      this.displayDataPreview();
      this.prepareData();
      this.setStatus('success', `Loaded ${data.length} samples. Ready to train.`);
    } catch (error) {
      console.error(error);
      this.setStatus('error', `Error: ${error.message}`);
    }
  }

  displayDataPreview() {
    const preview = this.elements.dataPreview;
    const sampleSize = Math.min(5, this.data.length);
    let html = '<table style="width: 100%; font-size: 11px; border-collapse: collapse;">';
    html += '<tr>';
    this.features.forEach(f => {
      html += `<td style="padding: 2px; border-bottom: 1px solid rgba(0,0,0,0.1);">
        <span style="font-weight: 600;">${f.substring(0, 4)}</span>
      </td>`;
    });
    html += '<td style="padding: 2px; border-bottom: 1px solid rgba(0,0,0,0.1);"><span style="font-weight: 600;">target</span></td></tr>';

    for (let i = 0; i < sampleSize; i++) {
      html += '<tr>';
      this.features.forEach(f => {
        html += `<td style="padding: 2px; border-bottom: 1px solid rgba(0,0,0,0.1);">
          ${this.data[i][f].toFixed(2)}
        </td>`;
      });
      html += `<td style="padding: 2px; border-bottom: 1px solid rgba(0,0,0,0.1);">
        ${this.data[i].virality}
      </td></tr>`;
    }

    html += `</table><div style="margin-top: 8px; color: rgba(0,0,0,0.5); font-size: 10px;">
      ... and ${this.data.length - sampleSize} more rows
    </div>`;
    preview.innerHTML = html;
  }

  prepareData() {
    // Extract features and target
    const X = this.data.map(row => this.features.map(f => row[f]));
    const y = this.data.map(row => row.virality);

    // Normalize features (mean=0, std=1)
    this.scaler.mean = [];
    this.scaler.std = [];

    for (let j = 0; j < this.features.length; j++) {
      const col = X.map(row => row[j]);
      const mean = col.reduce((a, b) => a + b, 0) / col.length;
      const variance = col.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / col.length;
      const std = Math.sqrt(variance);

      this.scaler.mean.push(mean);
      this.scaler.std.push(std || 1);

      for (let i = 0; i < X.length; i++) {
        X[i][j] = (X[i][j] - mean) / (std || 1);
      }
    }

    // Train-validation split (80-20)
    const splitIdx = Math.floor(this.data.length * 0.8);
    this.trainX = tf.tensor2d(X.slice(0, splitIdx));
    this.trainY = tf.tensor2d(y.slice(0, splitIdx), [splitIdx, 1]);
    this.valX = tf.tensor2d(X.slice(splitIdx));
    this.valY = tf.tensor2d(y.slice(splitIdx), [this.data.length - splitIdx, 1]);
  }

  buildModel() {
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [this.features.length],
          units: 128,
          activation: 'relu',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({
          units: 32,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid'
        })
      ]
    });

    const learningRate = parseFloat(this.elements.learningRate.value);
    const optimizer = tf.train.adam(learningRate);

    this.model.compile({
      optimizer: optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
  }

  async startTraining() {
    if (!this.data) {
      this.setStatus('error', 'Please upload data first');
      return;
    }

    if (this.isTraining) return;

    try {
      this.isTraining = true;
      this.elements.trainBtn.disabled = true;
      this.elements.stopBtn.disabled = false;
      this.elements.dataInput.disabled = true;
      this.setStatus('info', 'Building model...');

      // Reset history
      this.history = { loss: [], val_loss: [], acc: [], val_acc: [], auc: [] };

      // Build model
      this.buildModel();

      const epochs = parseInt(this.elements.epochs.value);
      const batchSize = parseInt(this.elements.batchSize.value);

      // Show progress UI
      this.elements.trainingProgress.style.display = 'block';
      this.elements.metricsPlaceholder.classList.add('hidden');
      this.elements.metricsContainer.classList.remove('hidden');
      this.elements.chartPlaceholder.classList.add('hidden');
      this.elements.chartContainer.classList.remove('hidden');

      this.setStatus('info', 'Training model...');

      // Training loop with manual epoch control for real-time updates
      for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
        // Training step
        const trainMetrics = await this.model.fit(this.trainX, this.trainY, {
          epochs: 1,
          batchSize: batchSize,
          verbose: 0,
          shuffle: true
        });

        // Validation step
        const valPredictions = this.model.predict(this.valX);
        const valPred = await valPredictions.data();
        valPredictions.dispose();

        const valTrue = await this.valY.data();
        const valMetrics = this.calculateMetrics(Array.from(valTrue), Array.from(valPred));

        // Store history
        this.history.loss.push(await trainMetrics.history.loss[0]);
        this.history.val_loss.push(valMetrics.loss);
        this.history.acc.push(await trainMetrics.history.acc[0]);
        this.history.val_acc.push(valMetrics.accuracy);
        this.history.auc.push(valMetrics.auc);

        // Update UI
        this.updateMetricsDisplay();
        this.drawCharts();

        // Update progress bar
        const progress = ((epoch + 1) / epochs) * 100;
        this.elements.progressBar.style.width = progress + '%';
        this.elements.epochDisplay.textContent = `${epoch + 1} / ${epochs}`;

        // Cleanup
        trainMetrics.dispose();

        // Allow UI to update
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      if (this.isTraining) {
        this.setStatus('success', 'Training completed!');
        this.elements.exportBtn.disabled = false;
        this.elements.exportInfo.textContent = 'Model ready for download';
      } else {
        this.setStatus('warning', 'Training stopped by user');
      }
    } catch (error) {
      console.error(error);
      this.setStatus('error', `Training failed: ${error.message}`);
    } finally {
      this.isTraining = false;
      this.elements.trainBtn.disabled = false;
      this.elements.stopBtn.disabled = true;
      this.elements.dataInput.disabled = false;
      this.elements.trainingProgress.style.display = 'none';
    }
  }

  stopTraining() {
    this.isTraining = false;
  }

  calculateMetrics(y_true, y_pred) {
    // Binary crossentropy loss
    let loss = 0;
    let correct = 0;
    let tp = 0, fp = 0, fn = 0, tn = 0;

    for (let i = 0; i < y_true.length; i++) {
      const true_val = y_true[i];
      const pred_prob = y_pred[i];
      const pred_class = pred_prob > 0.5 ? 1 : 0;

      // Loss
      const epsilon = 1e-7;
      const pred_clipped = Math.max(epsilon, Math.min(1 - epsilon, pred_prob));
      loss += -(true_val * Math.log(pred_clipped) + (1 - true_val) * Math.log(1 - pred_clipped));

      // Accuracy
      if (pred_class === true_val) correct++;

      // AUC components
      if (true_val === 1 && pred_class === 1) tp++;
      else if (true_val === 0 && pred_class === 1) fp++;
      else if (true_val === 1 && pred_class === 0) fn++;
      else if (true_val === 0 && pred_class === 0) tn++;
    }

    loss /= y_true.length;
    const accuracy = correct / y_true.length;

    // Simple AUC approximation (threshold-based)
    const tpr = tp / (tp + fn || 1);
    const fpr = fp / (fp + tn || 1);
    const auc = (1 + tpr - fpr) / 2; // Rough approximation

    return { loss, accuracy, auc };
  }

  updateMetricsDisplay() {
    const lastIdx = this.history.loss.length - 1;
    if (lastIdx < 0) return;

    this.elements.trainLoss.textContent = this.history.loss[lastIdx].toFixed(4);
    this.elements.valLoss.textContent = this.history.val_loss[lastIdx].toFixed(4);
    this.elements.trainAcc.textContent = (this.history.acc[lastIdx] * 100).toFixed(2) + '%';
    this.elements.valAcc.textContent = (this.history.val_acc[lastIdx] * 100).toFixed(2) + '%';
    this.elements.valAuc.textContent = this.history.auc[lastIdx].toFixed(4);
  }

  drawCharts() {
    // Loss chart
    this.drawChart(
      this.elements.lossCanvas,
      this.history.loss,
      this.history.val_loss,
      'Loss'
    );

    // Accuracy chart
    this.drawChart(
      this.elements.accCanvas,
      this.history.acc.map(v => v * 100),
      this.history.val_acc.map(v => v * 100),
      'Accuracy (%)'
    );
  }

  drawChart(canvas, trainData, valData, label) {
    const ctx = canvas.getContext('2d');
    const width = canvas.offsetWidth || 400;
    const height = canvas.offsetHeight || 240;

    canvas.width = width;
    canvas.height = height;

    if (trainData.length === 0) return;

    // Find min/max
    const allData = [...trainData, ...valData];
    const minVal = Math.min(...allData);
    const maxVal = Math.max(...allData);
    const range = maxVal - minVal || 1;
    const padding = 40;

    // Draw background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (i / 4) * (height - padding * 2);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Plot training data
    ctx.strokeStyle = '#2180a4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < trainData.length; i++) {
      const x = padding + (i / (trainData.length - 1 || 1)) * (width - padding * 2);
      const y = height - padding - ((trainData[i] - minVal) / range) * (height - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot validation data
    ctx.strokeStyle = '#ff5459';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let i = 0; i < valData.length; i++) {
      const x = padding + (i / (valData.length - 1 || 1)) * (width - padding * 2);
      const y = height - padding - ((valData[i] - minVal) / range) * (height - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw legend
    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#2180a4';
    ctx.fillText('Train', width - 140, 25);
    ctx.fillStyle = '#ff5459';
    ctx.fillText('Validation', width - 80, 25);
  }

  async exportModel() {
    if (!this.model) {
      this.setStatus('error', 'No model to export');
      return;
    }

    try {
      this.setStatus('info', 'Exporting model...');

      // Save model with metadata
      await this.model.save(
        'downloads://virality_model',
        {
          metadata: {
            features: this.features,
            scaler: this.scaler,
            history: this.history
          }
        }
      );

      this.setStatus('success', 'Model exported successfully!');
    } catch (error) {
      console.error(error);
      this.setStatus('error', `Export failed: ${error.message}`);
    }
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new ViralityPredictor();
});
