# Complaint Routing System Using DistilBERT

A machine learning system that automatically categorizes and routes student complaints to appropriate handlers using transformer-based text classification.

## Overview

The system uses DistilBERT, a lightweight BERT variant, to understand and classify student complaints into predefined categories and route them to appropriate handling departments.

## Architecture

### 1. Data Processing Layer
- **Data Structure**
  - Complaint text (input)
  - Category labels (target)
  - Handler mapping (routing)

- **Categories**
  ```python
  categories = [
      "Academic Issue",
      "Management/Hostel Issue",
      "Registration Problem",
      "Conduct-related Issue"
  ]
  ```

### 2. Model Architecture
- **Base Model**: DistilBERT
  - Pre-trained on general text
  - Fine-tuned for complaint classification
  - Lightweight and efficient

- **Components**
  1. Tokenizer (DistilBertTokenizerFast)
     - Converts text to tokens
     - Handles padding and truncation
     - Max sequence length: 128 tokens

  2. Classifier (TFDistilBertForSequenceClassification)
     - DistilBERT base uncased
     - Classification head
     - Multi-class output

### 3. Training Pipeline
1. **Data Preparation**
   - Text tokenization
   - Label encoding
   - Train-test split (80-20)

2. **Model Configuration**
   - Optimizer: Adam (lr=5e-5)
   - Loss: SparseCategoricalCrossentropy
   - Metric: Accuracy

3. **Training**
   - Batch size: 4
   - Epochs: 30
   - Dataset shuffling
   - TensorFlow data pipeline

### 4. Routing Logic
```python
routing_map = {
    "Academic Issue": "Faculty Mentor",
    "Registration Problem": "Academic Office",
    "Conduct-related Issue": "Student Welfare Dean",
    "Management/Hostel Issue": "Hostel/Admin Department"
}
```

## Usage

1. **Setup Environment**
   ```bash
   pip install tensorflow transformers pandas numpy scikit-learn
   ```

2. **Load Model**
   ```python
   from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

   tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
   model = TFDistilBertForSequenceClassification.from_pretrained(
       "distilbert-base-uncased", 
       num_labels=len(categories)
   )
   ```

3. **Route Complaint**
   ```python
   complaint = "My attendance has not been updated properly."
   result = route_complaint(complaint)
   print(f"Routed to: {result['Routed To']}")
   ```

## Training Results

- Model trained on synthetic dataset
- Evaluation metrics:
  - Classification report per category
  - Overall accuracy
  - Cross-validation scores

## System Requirements

- Python 3.7+
- TensorFlow 2.x
- Transformers library
- 4GB+ RAM
- GPU recommended but optional

## Performance Optimization

1. **Memory Efficiency**
   - Batch processing
   - Gradient accumulation
   - Model quantization

2. **Speed Optimization**
   - TensorFlow data pipeline
   - Prefetching
   - GPU acceleration

## Deployment Considerations

1. **Model Serving**
   - REST API endpoint
   - Batch prediction support
   - Error handling

2. **Monitoring**
   - Prediction confidence scores
   - Category distribution
   - Response times

## Future Improvements

1. **Model Enhancement**
   - Larger training dataset
   - Custom token handling
   - Multilingual support

2. **Feature Additions**
   - Priority classification
   - Response time estimation
   - Automatic response generation

3. **System Integration**
   - Email integration
   - Ticketing system connection
   - Dashboard visualization

## Limitations

- Limited to predefined categories
- Requires clear complaint text
- English language only
- No sentiment analysis

## License

MIT License - See LICENSE file for details
