from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np


def custom_tokenizer(text):
    """Custom tokenizer that splits text by commas."""
    return text.split(',')


def create_model():
    """Create a pipeline with TF-IDF vectorization and RandomForestClassifier."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    return model


def train_model(model, X_train, y_train):
    """Train the pipeline model with training data."""
    model.fit(X_train, y_train)


def save_model(model, model_filename):
    """Save the trained model to a file."""
    joblib.dump(model, model_filename)


def load_model(model_filename):
    """Load a model from a file."""
    return joblib.load(model_filename)


def predict_category(texts, model):
    """Predict categories for input texts."""
    return model.predict(texts)


def extract_important_features(tfidf_step, importances, n=10):
    """
    Extract the top n important features based on TF-IDF scores.

    :param tfidf_step: Fitted TF-IDF transformer.
    :param importances: Feature importances from the classifier.
    :param n: Number of top features to extract.
    :return: List of dictionaries with feature names and scores.
    """
    feature_names = np.array(tfidf_step.get_feature_names_out())
    indices = importances.argsort()[-n:][::-1]
    return [
        {"feature_name": feature_names[idx], "importance_score": importances[idx]}
        for idx in indices
    ]


def get_important_features_from_text(input_text):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Use TF-IDF to extract important features
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([input_text])  # Transform the input text
    feature_names = vectorizer.get_feature_names_out()  # Get words (features)
    scores = tfidf_matrix.toarray()[0]  # Get the TF-IDF scores for the words

    # Get the top 10 important features based on TF-IDF score
    sorted_indices = scores.argsort()[::-1]
    top_features = [
        {"feature_name": feature_names[idx], "importance_score": scores[idx]}
        for idx in sorted_indices[:10]
    ]
    return top_features
