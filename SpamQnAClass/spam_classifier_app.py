import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load the saved model
model = None
cv = None  # vectorizer

try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Determine expected feature count from model, if available
expected_features = getattr(model, "n_features_in_", None)

# Try to load a saved vectorizer first, otherwise build/fallback to TfidfVectorizer
vectorizer_path = Path("vectorizer.pkl")
if vectorizer_path.exists():
    try:
        cv = pickle.load(open(vectorizer_path, "rb"))
    except Exception:
        cv = None

# If no saved vectorizer, attempt to build from dataset.csv (if present), else will fit later on input
if cv is None:
    try:
        # prefer to match model's expected feature count if available
        default_max_features = expected_features if expected_features is not None else 3000
        tfidf = TfidfVectorizer(max_features=default_max_features)
        corpus_fit = None
        if Path("dataset.csv").exists():
            try:
                df_corpus = pd.read_csv("dataset.csv")
                # prefer an already preprocessed column if available
                if "transformed_text" in df_corpus.columns:
                    corpus_texts = df_corpus["transformed_text"].dropna().astype(str).tolist()
                elif "Message" in df_corpus.columns:
                    # apply same preprocessing used in notebook
                    def _transform_for_corpus(text):
                        text = str(text).lower()
                        tokens = nltk.word_tokenize(text)
                        y = [t for t in tokens if t.isalnum()]
                        y = [t for t in y if t not in stopwords.words('english') and t not in string.punctuation]
                        return " ".join([ps.stem(t) for t in y])
                    corpus_texts = df_corpus["Message"].dropna().astype(str).apply(_transform_for_corpus).tolist()
                else:
                    corpus_texts = None

                if corpus_texts:
                    # If model expects a specific number of features, create vectorizer accordingly
                    if expected_features is not None:
                        tfidf = TfidfVectorizer(max_features=expected_features)
                    tfidf.fit(corpus_texts)
                    cv = tfidf
                    corpus_fit = True
                    # persist fitted vectorizer for later runs
                    try:
                        pickle.dump(cv, open(vectorizer_path, "wb"))
                    except Exception:
                        pass
            except Exception:
                cv = None

        # leave cv as None if no dataset available; keep tfidf instance to fit later on user-upload or input
        if cv is None:
            # If we know expected_features, initialize tfidf accordingly so future fits use correct dimensionality
            if expected_features is not None:
                tfidf = TfidfVectorizer(max_features=expected_features)
            cv = tfidf  # keep instance to fit later on input
    except Exception:
        cv = None

def transform_text(text):
    # Same preprocessing as notebook
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

def predict_spam(text):
    # Transform text
    transformed_text = transform_text(text)
    
    # If vectorizer not fitted, fit on this single sample (best-effort) and warn user
    global cv
    if cv is None:
        raise RuntimeError("No vectorizer available. Please provide 'vectorizer.pkl' or dataset.csv in the app folder.")
    try:
        # If vectorizer hasn't been fitted yet, fit on input (best-effort fallback)
        if not hasattr(cv, "vocabulary_") or getattr(cv, "vocabulary_") is None:
            st.warning("No pre-trained vectorizer found. Fitting TF-IDF on provided input (results may be unreliable).")
            # If model expects features, initialize cv accordingly before fitting
            if expected_features is not None and getattr(cv, "max_features", None) != expected_features:
                cv = TfidfVectorizer(max_features=expected_features)
            cv.fit([transformed_text])
            # persist fitted vectorizer
            try:
                pickle.dump(cv, open(vectorizer_path, "wb"))
            except Exception:
                pass

        vector_input = cv.transform([transformed_text])
    except Exception as e:
        raise RuntimeError(f"Vectorization failed: {e}")

    # Convert to dense numpy array for manipulation
    try:
        X = vector_input.toarray()
    except Exception:
        # fallback: convert sparse to dense safely
        X = np.array(vector_input.todense())

    # Align feature dimension with model expectation if possible
    expected = getattr(model, "n_features_in_", None)
    if expected is not None:
        if X.shape[1] < expected:
            # pad with zeros to match expected feature count
            X_padded = np.zeros((X.shape[0], expected), dtype=X.dtype)
            X_padded[:, :X.shape[1]] = X
            X = X_padded
            st.warning(f"Vector has {vector_input.shape[1]} features; padded to model's expected {expected} features. Predictions may be unreliable without the original vectorizer.")
        elif X.shape[1] > expected:
            # truncate to expected size
            X = X[:, :expected]
            st.warning(f"Vector has {vector_input.shape[1]} features; truncated to model's expected {expected} features.")
    else:
        # No info on expected features; proceed as-is
        pass

    # Predict
    try:
        result = model.predict(X)[0]
        # handle models that may not implement predict_proba
        try:
            prob = model.predict_proba(X)[0]
        except Exception:
            prob = [0.0, 0.0] if hasattr(model, "classes_") else [0.0]
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    
    return result, prob, transformed_text

# Streamlit UI
st.title('Review Spam Classifier')

# Provide optional corpus upload to build TF-IDF if vectorizer not available
if vectorizer_path.exists() is False:
    st.info("No vectorizer.pkl found. You can upload a corpus CSV (with column 'Message' or 'transformed_text') to build TF-IDF.")
    corpus_file = st.file_uploader("Upload corpus CSV to build TF-IDF (optional)", type=["csv"])
    if corpus_file is not None:
        try:
            df_up = pd.read_csv(corpus_file)
            if "transformed_text" in df_up.columns:
                texts_for_fit = df_up["transformed_text"].dropna().astype(str).tolist()
            elif "Message" in df_up.columns:
                texts_for_fit = df_up["Message"].dropna().astype(str).apply(transform_text).tolist()
            else:
                texts_for_fit = df_up.iloc[:,0].dropna().astype(str).apply(transform_text).tolist()

            # If model expects a specific feature count, ensure vectorizer uses it
            if expected_features is not None:
                cv = TfidfVectorizer(max_features=expected_features)
            else:
                cv = TfidfVectorizer(max_features=3000)

            cv.fit(texts_for_fit)
            # persist the vectorizer for future runs
            try:
                pickle.dump(cv, open(vectorizer_path, "wb"))
                st.success("TF-IDF vectorizer fitted on uploaded corpus and saved as vectorizer.pkl.")
            except Exception:
                st.success("TF-IDF vectorizer fitted on uploaded corpus.")
        except Exception as e:
            st.error(f"Failed to build TF-IDF from uploaded corpus: {e}")

# Text input
user_input = st.text_area('Enter the review text:', height=100)

if st.button('Check for Spam'):
    if not user_input:
        st.warning('Please enter some text')
    else:
        try:
            with st.spinner('Analyzing...'):
                result, prob, transformed_text = predict_spam(user_input)
                
                # Display result
                st.header('Results')
                if result == 1:
                    st.error('Spam Review Detected!')
                    confidence = prob[1] if len(prob) > 1 else prob[0]
                else:
                    st.success('Legitimate Review!')
                    confidence = prob[0] if len(prob) > 0 else 0.0
                
                st.metric('Confidence', f'{confidence:.2%}')
                
                # Show preprocessing details in expander
                with st.expander('See preprocessing details'):
                    st.write('Original text:', user_input)
                    st.write('Cleaned and transformed text:', transformed_text)
                    st.write('Probability distribution:')
                    # format probability dict if possible
                    if isinstance(prob, (list, tuple)) and len(prob) >= 2:
                        st.write({
                            'Ham (legitimate)': f'{prob[0]:.4f}',
                            'Spam': f'{prob[1]:.4f}'
                        })
                    else:
                        st.write(prob)
        except Exception as e:
            st.error(f"Error: {e}")

# Add helpful instructions in sidebar
st.sidebar.title('About')
st.sidebar.info(
    'This app uses a machine learning model trained on review data '
    'to detect spam reviews. If you do not have vectorizer.pkl, the app will attempt '
    'to build a TF-IDF vectorizer from dataset.csv (if present) or from an uploaded corpus. '
    'Predictions may be unreliable if the vectorizer does not match the one used during training.'
)
st.sidebar.markdown('### Instructions')
st.sidebar.markdown(
    '1. If available, place vectorizer.pkl in the app folder for best results.\n'
    '2. Optionally upload a corpus CSV to build TF-IDF (column: Message or transformed_text).\n'
    '3. Enter review text and click "Check for Spam".'
)
