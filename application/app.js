/**
 * VideoSuccessPredictor - Main application class
 * Uses TensorFlow.js to predict video success based on features
 */
import { DataLoader } from './data-loader.js';

class VideoSuccessPredictor {
    constructor() {
        this.model = null;
        this.dataLoader = new DataLoader();
        this.isTraining = false;
        this.trainingHistory = [];
        this.testResults = null;
        
        // UI elements
        this.ui = {
            trainingProgress: document.getElementById('trainingProgress'),
            trainingStatus: document.getElementById('trainingStatus'),
            trainingLog: document.getElementById('trainingLog'),
            metricsDisplay: document.getElementById('metricsDisplay'),
            predictButton: document.getElementById('predictButton'),
            sampleInputs: document.querySelectorAll('.sample-input'),
            predictionResult: document.getElementById('predictionResult'),
            modelStatus: document.getElementById('modelStatus'),
            trainTime: document.getElementById('trainTime'),
            testTime: document.getElementById('testTime')
        };
        
        this.initializeEventListeners();
    }

    /**
     * Initialize event listeners for UI
     */
    initializeEventListeners() {
        if (this.ui.predictButton) {
            this.ui.predictButton.addEventListener('click', () => this.predictFromUI());
        }
        
        // Update prediction when sample inputs change
        this.ui.sampleInputs.forEach(input => {
            input.addEventListener('change', () => this.updatePrediction());
            input.addEventListener('input', () => this.updatePrediction());
        });
    }

    /**
     * Initialize and build the neural network model
     */
    buildModel(inputShape) {
        try {
            this.model = tf.sequential();
            
            // Input layer + first hidden layer
            this.model.add(tf.layers.dense({
                units: 64,
                activation: 'relu',
                inputShape: [inputShape]
            }));
            
            // Dropout for regularization
            this.model.add(tf.layers.dropout({ rate: 0.2 }));
            
            // Second hidden layer
            this.model.add(tf.layers.dense({
                units: 32,
                activation: 'relu'
            }));
            
            // Third hidden layer
            this.model.add(tf.layers.dense({
                units: 16,
                activation: 'relu'
            }));
            
            // Output layer (single neuron for regression)
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'linear'
            }));
            
            // Compile the model
            this.model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'meanSquaredError',
                metrics: ['mse', 'mae']
            });
            
            console.log('Model built successfully');
            this.updateUIStatus('Model built', 'success');
            
        } catch (error) {
            console.error('Error building model:', error);
            this.updateUIStatus('Error building model', 'error');
            throw error;
        }
    }

    /**
     * Train the model
     * @param {Array} trainFeatures - Training features
     * @param {Array} trainLabels - Training labels
     * @param {Array} testFeatures - Testing features (optional)
     * @param {Array} testLabels - Testing labels (optional)
     */
    async trainModel(trainFeatures, trainLabels, testFeatures = null, testLabels = null) {
        if (this.isTraining) {
            console.warn('Training already in progress');
            return;
        }
        
        try {
            this.isTraining = true;
            this.trainingHistory = [];
            this.updateUIStatus('Training started...', 'info');
            
            // Convert data to tensors
            const trainFeatureTensor = tf.tensor2d(trainFeatures);
            const trainLabelTensor = tf.tensor1d(trainLabels);
            
            let validationData = null;
            if (testFeatures && testLabels) {
                validationData = [
                    tf.tensor2d(testFeatures),
                    tf.tensor1d(testLabels)
                ];
            }
            
            const startTime = performance.now();
            
            // Train the model
            const history = await this.model.fit(trainFeatureTensor, trainLabelTensor, {
                epochs: 100,
                batchSize: 32,
                validationSplit: 0.2,
                validationData: validationData,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.trainingHistory.push({
                            epoch: epoch + 1,
                            loss: logs.loss,
                            valLoss: logs.val_loss,
                            mae: logs.mae,
                            valMae: logs.val_mae
                        });
                        
                        this.updateTrainingProgress(epoch + 1, logs);
                        
                        // Dispose intermediate tensors to prevent memory leaks
                        tf.dispose([trainFeatureTensor, trainLabelTensor]);
                        if (validationData) {
                            tf.dispose(validationData);
                        }
                    },
                    onTrainEnd: () => {
                        const endTime = performance.now();
                        const trainingTime = ((endTime - startTime) / 1000).toFixed(2);
                        this.ui.trainTime.textContent = trainingTime;
                        this.updateUIStatus('Training completed', 'success');
                        this.isTraining = false;
                        
                        // Test the model immediately after training
                        if (testFeatures && testLabels) {
                            this.testModel(testFeatures, testLabels);
                        }
                    }
                }
            });
            
            // Cleanup tensors
            trainFeatureTensor.dispose();
            trainLabelTensor.dispose();
            if (validationData) {
                validationData[0].dispose();
                validationData[1].dispose();
            }
            
            return history;
            
        } catch (error) {
            console.error('Error training model:', error);
            this.updateUIStatus('Training failed', 'error');
            this.isTraining = false;
            throw error;
        }
    }

    /**
     * Test the model on unseen data
     * @param {Array} testFeatures - Test features
     * @param {Array} testLabels - Test labels
     */
    async testModel(testFeatures, testLabels) {
        try {
            this.updateUIStatus('Testing model...', 'info');
            const startTime = performance.now();
            
            const testFeatureTensor = tf.tensor2d(testFeatures);
            const testLabelTensor = tf.tensor1d(testLabels);
            
            // Get predictions
            const predictionsTensor = this.model.predict(testFeatureTensor);
            const predictions = await predictionsTensor.data();
            
            // Calculate metrics
            const mse = tf.metrics.meanSquaredError(testLabelTensor, predictionsTensor);
            const mae = tf.metrics.meanAbsoluteError(testLabelTensor, predictionsTensor);
            
            const mseValue = (await mse.data())[0];
            const maeValue = (await mae.data())[0];
            
            // Calculate R-squared
            const ssRes = tf.sum(tf.square(tf.sub(testLabelTensor, predictionsTensor)));
            const meanLabel = tf.mean(testLabelTensor);
            const ssTot = tf.sum(tf.square(tf.sub(testLabelTensor, meanLabel)));
            const r2 = 1 - (await ssRes.data())[0] / (await ssTot.data())[0];
            
            const endTime = performance.now();
            const testTime = ((endTime - startTime) / 1000).toFixed(2);
            this.ui.testTime.textContent = testTime;
            
            this.testResults = {
                mse: mseValue,
                mae: maeValue,
                r2: r2,
                predictions: Array.from(predictions),
                actual: testLabels,
                testTime: testTime
            };
            
            this.displayMetrics();
            this.updateUIStatus('Testing completed', 'success');
            
            // Cleanup tensors
            tf.dispose([testFeatureTensor, testLabelTensor, predictionsTensor, mse, mae, ssRes, meanLabel, ssTot]);
            
        } catch (error) {
            console.error('Error testing model:', error);
            this.updateUIStatus('Testing failed', 'error');
            throw error;
        }
    }

    /**
     * Make prediction from UI inputs
     */
    async predictFromUI() {
        if (!this.model) {
            alert('Please train the model first');
            return;
        }
        
        try {
            // Collect input values
            const inputs = [];
            this.ui.sampleInputs.forEach(input => {
                let value = parseFloat(input.value);
                if (isNaN(value)) {
                    value = 0;
                }
                inputs.push(value);
            });
            
            // Normalize inputs (simplified - in real app, use same normalizers as training)
            const normalizedInputs = inputs.map((val, idx) => {
                // Simple normalization - in production, use the same normalizers as training data
                return Math.min(Math.max(val / 100, 0), 1);
            });
            
            // Make prediction
            const inputTensor = tf.tensor2d([normalizedInputs]);
            const predictionTensor = this.model.predict(inputTensor);
            const prediction = (await predictionTensor.data())[0];
            
            // Inverse transform prediction
            const viewsPredicted = this.dataLoader.inverseTransformLabels([prediction])[0];
            
            // Display result
            this.ui.predictionResult.innerHTML = `
                <div class="alert alert-success">
                    <h4>Predicted Total Views:</h4>
                    <p class="display-4">${Math.round(viewsPredicted).toLocaleString()}</p>
                    <small>Based on provided video characteristics</small>
                </div>
            `;
            
            // Cleanup
            inputTensor.dispose();
            predictionTensor.dispose();
            
        } catch (error) {
            console.error('Error making prediction:', error);
            this.ui.predictionResult.innerHTML = `
                <div class="alert alert-danger">
                    Error making prediction: ${error.message}
                </div>
            `;
        }
    }

    /**
     * Update prediction when inputs change (debounced)
     */
    updatePrediction() {
        if (this.predictDebounceTimer) {
            clearTimeout(this.predictDebounceTimer);
        }
        
        this.predictDebounceTimer = setTimeout(() => {
            if (this.model) {
                this.predictFromUI();
            }
        }, 500);
    }

    /**
     * Update UI with training progress
     * @param {number} epoch - Current epoch
     * @param {Object} logs - Training logs
     */
    updateTrainingProgress(epoch, logs) {
        if (!this.ui.trainingProgress || !this.ui.trainingStatus) return;
        
        this.ui.trainingProgress.value = epoch;
        this.ui.trainingStatus.textContent = `Epoch ${epoch}/100 - Loss: ${logs.loss?.toFixed(4)}`;
        
        // Add to log
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.textContent = `Epoch ${epoch}: Loss=${logs.loss?.toFixed(4)}, Val Loss=${logs.val_loss?.toFixed(4)}`;
        this.ui.trainingLog.prepend(logEntry);
        
        // Keep log manageable
        if (this.ui.trainingLog.children.length > 10) {
            this.ui.trainingLog.removeChild(this.ui.trainingLog.lastChild);
        }
    }

    /**
     * Display model metrics
     */
    displayMetrics() {
        if (!this.testResults || !this.ui.metricsDisplay) return;
        
        const { mse, mae, r2 } = this.testResults;
        
        this.ui.metricsDisplay.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <h5>Mean Squared Error</h5>
                    <p class="metric-value">${mse.toFixed(6)}</p>
                    <small>Lower is better</small>
                </div>
                <div class="metric-card">
                    <h5>Mean Absolute Error</h5>
                    <p class="metric-value">${mae.toFixed(6)}</p>
                    <small>Lower is better</small>
                </div>
                <div class="metric-card">
                    <h5>R-squared</h5>
                    <p class="metric-value">${r2.toFixed(4)}</p>
                    <small>Higher is better (0-1)</small>
                </div>
                <div class="metric-card">
                    <h5>Test Time</h5>
                    <p class="metric-value">${this.testResults.testTime}s</p>
                    <small>Seconds to test</small>
                </div>
            </div>
        `;
    }

    /**
     * Update UI status
     * @param {string} message - Status message
     * @param {string} type - Message type (success, error, info)
     */
    updateUIStatus(message, type = 'info') {
        if (!this.ui.modelStatus) return;
        
        this.ui.modelStatus.textContent = message;
        this.ui.modelStatus.className = `status-${type}`;
        
        // Auto-clear success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (this.ui.modelStatus.textContent === message) {
                    this.ui.modelStatus.textContent = 'Ready';
                    this.ui.modelStatus.className = 'status-ready';
                }
            }, 5000);
        }
    }

    /**
     * Initialize and run the application
     */
    async initialize() {
        try {
            this.updateUIStatus('Loading data...', 'info');
            
            // Load and preprocess data
            await this.dataLoader.loadCSV('data.csv');
            const { features, labels } = this.dataLoader.preprocessData();
            
            // Split data
            const { trainFeatures, trainLabels, testFeatures, testLabels } = 
                this.dataLoader.trainTestSplit(0.2);
            
            // Build model
            this.buildModel(trainFeatures[0].length);
            
            // Train model
            this.updateUIStatus('Starting training...', 'info');
            await this.trainModel(trainFeatures, trainLabels, testFeatures, testLabels);
            
            // Enable prediction button
            if (this.ui.predictButton) {
                this.ui.predictButton.disabled = false;
                this.ui.predictButton.textContent = 'Predict Views';
            }
            
        } catch (error) {
            console.error('Application initialization failed:', error);
            this.updateUIStatus(`Initialization failed: ${error.message}`, 'error');
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        if (this.dataLoader) {
            this.dataLoader.dispose();
        }
        
        this.trainingHistory = [];
        this.testResults = null;
        this.isTraining = false;
        
        console.log('VideoSuccessPredictor disposed');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new VideoSuccessPredictor();
    window.videoPredictor = app; // Make available globally for debugging
    
    // Start the application
    app.initialize().catch(error => {
        console.error('Failed to initialize application:', error);
    });
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.dispose();
    });
});
