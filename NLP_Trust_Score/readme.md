# Trust Score Sentiment Analysis System

A deep learning system that analyzes sentiment in company reviews to generate trust scores using Universal Sentence Encoder and BiLSTM architecture.

## Architecture Overview

### 1. Data Processing Pipeline
- **Text Preprocessing**
  - Emoji removal
  - Non-ASCII character filtering
  - Number normalization
  - URL removal
  - Special character handling

- **Data Structure**
  ```python
  {
    'Review': str,      # Review text
    'Rating': int,      # 1-5 rating
    'Id': int          # Unique identifier
  }
  ```

### 2. Model Architecture

#### Core Components
1. **Universal Sentence Encoder**
   - Pre-trained encoder from TensorFlow Hub
   - Generates 512-dimensional embeddings
   - Fixed (non-trainable) weights
   
2. **BiDirectional LSTM**
   - 128 units per direction
   - HeNormal initialization
   - Processes temporal patterns

3. **Classification Head**
   - Dropout (0.25) for regularization
   - Dense layer (64 units, ReLU)
   - Output layer (5 classes, Softmax)

#### Training Configuration
- Batch Size: 32
- Epochs: 10 (with early stopping)
- Loss: Categorical Crossentropy with label smoothing (0.1)
- Optimizer: Adam (lr=0.001)
- Learning rate reduction on plateau

### 3. Data Pipeline

1. **Data Loading**
   - CSV file ingestion
   - Train/validation split (80/20)
   - Stratified sampling by rating

2. **TensorFlow Data Pipeline**
   - tf.data.Dataset implementation
   - Batching and prefetching
   - Optional caching
   - Dynamic shuffling

### 4. Training Features

- **Early Stopping**
  - Monitors validation loss
  - Patience: 4 epochs
  - Restores best weights

- **Learning Rate Schedule**
  - Reduces on plateau
  - Patience: 2 epochs
  - Reduction factor: 0.1

### 5. Performance Metrics

1. **Core Metrics**
   - Accuracy
   - Top-2 Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-Score (weighted)
   - Matthews Correlation Coefficient

2. **Visualization**
   - Loss curves
   - Accuracy curves
   - Confusion matrix
   - Rating distributions

### 6. Deployment Workflow

1. **Model Loading**
   ```python
   encoder = hub.KerasLayer('universal-sentence-encoder')
   model = build_baseline_model()
   ```

2. **Inference**
   ```python
   text = preprocess_text(input_text)
   probabilities = model.predict([text])
   rating = tf.argmax(probabilities) + 1
   ```

## Requirements

- TensorFlow 2.x
- TensorFlow Hub
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Performance Characteristics

- Training speed: ~100ms/step
- Memory usage: ~2GB RAM
- GPU acceleration supported
- Inference time: ~50ms/sample

## Limitations

1. **Input Constraints**
   - English text only
   - Maximum sequence length: 128 tokens
   - Requires clean text input

2. **Model Constraints**
   - Fixed rating scale (1-5)
   - Single-label classification
   - No multi-modal support

## Future Improvements

1. **Architecture**
   - Transformer-based models
   - Multi-aspect sentiment analysis
   - Cross-lingual support

2. **Features**
   - Attention visualization
   - Confidence scoring
   - Batch prediction API

3. **Training**
   - Mixed precision training
   - Gradient accumulation
   - Model distillation

## License

MIT License - See LICENSE file for details
