/**
 * DataLoader class - handles loading and preprocessing of video data from CSV
 * Uses ES6 module syntax with client-side TensorFlow.js
 */
export class DataLoader {
    constructor() {
        this.data = null;
        this.features = null;
        this.labels = null;
        this.featureNames = ['duration_sec', 'hook_strength_score', 'niche', 'views_first_hour', 
                            'retention_rate', 'first_3_sec_engagement', 'music_type', 'upload_month'];
        this.labelName = 'views_total';
        this.normalizers = {};
        this.labelNormalizer = {};
    }

    /**
     * Load data from CSV file
     * @param {string} filePath - Path to CSV file
     * @returns {Promise<Array>} - Promise resolving to parsed data
     */
    async loadCSV(filePath) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`Failed to load CSV: ${response.status} ${response.statusText}`);
            }
            
            const csvText = await response.text();
            const rows = csvText.trim().split('\n');
            const headers = rows[0].split(',');
            
            // Validate required columns
            const requiredColumns = [...this.featureNames, this.labelName];
            for (const col of requiredColumns) {
                if (!headers.includes(col)) {
                    throw new Error(`Missing required column: ${col}`);
                }
            }
            
            // Parse data
            this.data = [];
            for (let i = 1; i < rows.length; i++) {
                const values = rows[i].split(',');
                const row = {};
                
                for (let j = 0; j < headers.length; j++) {
                    const header = headers[j].trim();
                    row[header] = parseFloat(values[j].trim());
                }
                
                // Validate row has all required fields
                if (Object.keys(row).length === headers.length) {
                    this.data.push(row);
                }
            }
            
            if (this.data.length === 0) {
                throw new Error('No valid data found in CSV');
            }
            
            console.log(`Loaded ${this.data.length} rows of data`);
            return this.data;
            
        } catch (error) {
            console.error('Error loading CSV:', error);
            throw error;
        }
    }

    /**
     * Preprocess data: normalize features and label
     * @returns {Object} - Preprocessed features and labels
     */
    preprocessData() {
        if (!this.data || this.data.length === 0) {
            throw new Error('No data loaded. Call loadCSV first.');
        }
        
        // Extract features and labels
        const rawFeatures = [];
        const rawLabels = [];
        
        for (const row of this.data) {
            const featureRow = this.featureNames.map(name => row[name]);
            rawFeatures.push(featureRow);
            rawLabels.push(row[this.labelName]);
        }
        
        // Normalize features
        this.features = this.normalizeFeatures(rawFeatures);
        
        // Log-transform and normalize labels (views are highly skewed)
        this.labels = this.normalizeLabels(rawLabels);
        
        console.log('Data preprocessing complete');
        console.log(`Features shape: ${this.features.length} x ${this.features[0].length}`);
        console.log(`Labels shape: ${this.labels.length}`);
        
        return {
            features: this.features,
            labels: this.labels,
            normalizers: this.normalizers,
            labelNormalizer: this.labelNormalizer
        };
    }

    /**
     * Normalize features to [0, 1] range
     * @param {Array} rawFeatures - Raw feature data
     * @returns {Array} - Normalized features
     */
    normalizeFeatures(rawFeatures) {
        const numFeatures = rawFeatures[0].length;
        const normalized = [];
        
        // Calculate min and max for each feature
        for (let i = 0; i < numFeatures; i++) {
            const values = rawFeatures.map(row => row[i]);
            this.normalizers[i] = {
                min: Math.min(...values),
                max: Math.max(...values)
            };
        }
        
        // Apply normalization
        for (const row of rawFeatures) {
            const normalizedRow = [];
            for (let i = 0; i < numFeatures; i++) {
                const { min, max } = this.normalizers[i];
                if (max === min) {
                    normalizedRow.push(0); // Avoid division by zero
                } else {
                    normalizedRow.push((row[i] - min) / (max - min));
                }
            }
            normalized.push(normalizedRow);
        }
        
        return normalized;
    }

    /**
     * Log-transform and normalize labels
     * @param {Array} rawLabels - Raw label data
     * @returns {Array} - Normalized labels
     */
    normalizeLabels(rawLabels) {
        // Apply log transformation to handle skewed distribution
        const logLabels = rawLabels.map(val => Math.log10(val + 1)); // +1 to avoid log(0)
        
        // Normalize to [0, 1]
        const min = Math.min(...logLabels);
        const max = Math.max(...logLabels);
        
        this.labelNormalizer = {
            min,
            max,
            originalMin: Math.min(...rawLabels),
            originalMax: Math.max(...rawLabels)
        };
        
        return logLabels.map(val => (val - min) / (max - min));
    }

    /**
     * Inverse transform normalized predictions back to original scale
     * @param {Array} predictions - Normalized predictions
     * @returns {Array} - Predictions in original scale
     */
    inverseTransformLabels(predictions) {
        if (!this.labelNormalizer.min && this.labelNormalizer.min !== 0) {
            throw new Error('Label normalizer not initialized');
        }
        
        const { min, max } = this.labelNormalizer;
        const logPredictions = predictions.map(p => p * (max - min) + min);
        
        // Reverse log transformation: 10^x - 1
        return logPredictions.map(p => Math.pow(10, p) - 1);
    }

    /**
     * Split data into training and testing sets
     * @param {number} testSize - Proportion of data for testing (default: 0.2)
     * @returns {Object} - Split datasets
     */
    trainTestSplit(testSize = 0.2) {
        if (!this.features || !this.labels) {
            throw new Error('Data not preprocessed. Call preprocessData first.');
        }
        
        const shuffledIndices = Array.from({length: this.features.length}, (_, i) => i);
        for (let i = shuffledIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
        }
        
        const splitIdx = Math.floor(this.features.length * (1 - testSize));
        
        const trainFeatures = [];
        const trainLabels = [];
        const testFeatures = [];
        const testLabels = [];
        
        for (let i = 0; i < shuffledIndices.length; i++) {
            const idx = shuffledIndices[i];
            if (i < splitIdx) {
                trainFeatures.push(this.features[idx]);
                trainLabels.push(this.labels[idx]);
            } else {
                testFeatures.push(this.features[idx]);
                testLabels.push(this.labels[idx]);
            }
        }
        
        console.log(`Split data: ${trainFeatures.length} training, ${testFeatures.length} testing`);
        
        return {
            trainFeatures,
            trainLabels,
            testFeatures,
            testLabels
        };
    }

    /**
     * Clean up memory
     */
    dispose() {
        this.data = null;
        this.features = null;
        this.labels = null;
        this.normalizers = {};
        this.labelNormalizer = {};
    }
}
