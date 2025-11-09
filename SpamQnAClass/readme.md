# QnA Review Spam Classification

This project implements a text classification pipeline to detect spam in Q&A / review messages. The notebook walks through dataset inspection, preprocessing, feature extraction, model experimentation, evaluation, and final model export.

Summary of architecture and pipeline
1. Data ingestion
   - Load dataset (CSV) into a pandas DataFrame.
   - Inspect schema, check for nulls, duplicates, and basic statistics.

2. Exploratory Data Analysis (EDA)
   - Class distribution analysis (spam vs ham).
   - Text statistics: number of characters, words, sentences.
   - Visualizations: histograms, pairplots, correlation heatmaps, word clouds for spam/ham.

3. Text preprocessing
   - Lowercasing and tokenization.
   - Removal of non-alphanumeric tokens and punctuation.
   - Stopword removal (NLTK stopwords).
   - Stemming (PorterStemmer) applied to reduce words to root form.
   - Generate derived features: num_characters, num_words, num_sentence.

4. Feature engineering / vectorization
   - CountVectorizer (bag-of-words) baseline.
   - TfidfVectorizer for improved text weighting; max_features tuning (e.g. 3000).
   - Numerical features (if used) can be concatenated with text vectors.

5. Train / test split
   - Typical split 80/20 (train_test_split).
   - Stratify by target where appropriate to preserve class distribution.

6. Model experimentation
   - Tried algorithms:
     - Naive Bayes variants: GaussianNB, MultinomialNB, BernoulliNB
     - Linear models: LogisticRegression
     - Kernel-based: SVC
     - Tree and ensemble methods: DecisionTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, Bagging
     - K-Nearest Neighbors
   - A helper function trains a classifier and returns accuracy & precision for comparison.

7. Model selection and tuning
   - Use TF-IDF + MultinomialNB or ExtraTreesClassifier (based on notebook results).
   - Tune TF-IDF max_features and model hyperparameters for precision (precision prioritized for spam detection).
   - Compare algorithms using accuracy, precision, confusion matrix; visualize performance.

8. Final model training and persistence
   - Retrain best model on selected features (full training set).
   - Evaluate final model on test set.
   - Persist model using pickle (`model.pkl`) for deployment.

9. Evaluation metrics and diagnostics
   - Confusion matrix to inspect false positives / false negatives.
   - Accuracy, precision, recall, F1-score (precision emphasized for spam).
   - Visual analyses: word clouds and feature distributions to interpret model behavior.

10. Deployment considerations
    - Preprocessing must be identical between training and inference (tokenization, stemming, TF-IDF vectorizer).
    - Save both trained model and TF-IDF vectorizer (or pipeline) for inference.
    - Provide a small API wrapper (Flask/FastAPI) that:
      - Accepts raw text
      - Runs preprocessing -> vectorizer -> model.predict
      - Returns label and confidence score
    - Consider thresholds for classifying low-confidence predictions.

11. Performance & improvements
    - Address class imbalance with oversampling (SMOTE) or class-weighted models if needed.
    - Use cross-validation for robust model selection.
    - Try modern embeddings (Word2Vec/GloVe/BERT) and simple neural classifiers for improved recall/precision trade-offs.
    - Model compression or quantization for lightweight deployment.

12. Requirements
    - Python 3.7+
    - pandas, numpy, matplotlib, seaborn
    - scikit-learn
    - nltk
    - wordcloud
    - pickle (stdlib)
    - Optional: transformers / tensorflow if experimenting with deep models

13. Quick usage (inference)
    - Load saved vectorizer and model:
      - vectorizer = pickle.load(open("tfidf.pkl","rb"))
      - model = pickle.load(open("model.pkl","rb"))
    - Preprocess text using same transform_text() function from notebook.
    - Transform and predict:
      - X = vectorizer.transform([processed_text])
      - label = model.predict(X)

Notes and best practices
- Keep preprocessing code in a reusable module or pipeline (ensures inference parity).
- Save the TF-IDF vectorizer along with the model for deterministic results.
- Monitor false-positive rate in production: spam false positives can degrade user experience.
- Periodically retrain with new labeled examples to adapt to evolving spam patterns.
