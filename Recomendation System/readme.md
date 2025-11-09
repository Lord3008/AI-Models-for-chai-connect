# Skill-Based Freelancer Recommendation System

A machine learning-based recommendation system that matches clients with freelancers based on skills, project descriptions, and ratings.

## Overview

This system uses multiple factors to generate accurate freelancer recommendations:
- Text similarity between project descriptions and portfolios
- Skill matching using exact and fuzzy comparison
- Freelancer ratings and performance metrics

## Features

- **Text Understanding**: Uses BERT embeddings for semantic comprehension
- **Skill Matching**: Supports both exact and fuzzy skill matching
- **Configurable Weights**: Adjustable importance of different factors
- **GPU Acceleration**: Utilizes GPU when available for faster processing
- **Batch Processing**: Efficient handling of multiple recommendations
- **Rating Integration**: Considers freelancer performance history

## System Architecture

### 1. Data Layer
- **Synthetic Data Generation**
  - Creates realistic test data for clients, freelancers, and reviews
  - Uses Faker library for natural text generation
  - Maintains consistent skill universe

- **Data Structure**
  ```json
  {
    "clients": [{project_description, skills_required, ...}],
    "freelancers": [{skills, portfolio_text, avg_rating, ...}],
    "reviews": [{review_text, rating, ...}]
  }
  ```

### 2. Processing Layer
- **Text Preprocessing**
  - Lowercase conversion
  - Special character removal
  - Optional stopword removal
  - Lemmatization

- **Text Embedding**
  - BERT-based semantic encoding
  - 768-dimensional vectors
  - Batched processing for efficiency

- **Skill Processing**
  - One-hot encoding option
  - Set-based comparison
  - Fuzzy matching support
  - Jaccard similarity computation

### 3. Scoring Layer
- **Component Weights**
  - Text Similarity: 40%
  - Skill Match: 40%
  - Rating Score: 20%

- **Score Normalization**
  - All components normalized to [0,1]
  - Weighted combination for final score

## Usage

1. **Setup**
   ```python
   pip install -r requirements.txt
   ```

2. **Generate Test Data**
   ```python
   # Run data generation script
   python generate_synthetic_data.py
   ```

3. **Process Freelancer Data**
   ```python
   # Generate and save freelancer embeddings
   python process_freelancers.py
   ```

4. **Get Recommendations**
   ```python
   # For a new client
   client_data = {
       "project_description": "Build a React website",
       "skills_required": ["React", "JavaScript"]
   }
   scores = compute_final_score(client_data)
   ```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NLTK
- NumPy
- FuzzyWuzzy

## Performance Considerations

- Embeddings are cached for reuse
- GPU acceleration when available
- Batch processing for efficiency
- Optimized similarity calculations
- Pre-computed freelancer features

## Future Improvements

1. **Content Enhancement**
   - Portfolio analysis
   - Project history integration
   - Client feedback processing

2. **Algorithm Refinement**
   - Dynamic weight adjustment
   - Contextual skill matching
   - Time-based feature decay

3. **System Optimization**
   - Distributed processing
   - Real-time updates
   - Embedding compression

## License

MIT License - See LICENSE file for details
