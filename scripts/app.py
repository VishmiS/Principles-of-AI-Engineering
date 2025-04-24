from flask import Flask, request, jsonify
from model import load_model, get_important_features_from_text
from preprocessing import preprocess_text
import uuid
from db import init_db, get_db_connection
import datetime
from flask_cors import CORS
from langdetect import detect, DetectorFactory
from prometheus_client import Counter, Gauge, Summary, generate_latest, REGISTRY
from flask import Response
import os
from lime.lime_text import LimeTextExplainer
import numpy as np

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5000"}})  # Allow only specific origins

# Load the pre-trained model
model_path = os.path.join(os.getcwd(), 'random_forest_model.pkl')
model = load_model(model_path)
print("Model loaded successfully!")

# Initialize the database
init_db()

# Ensure consistent results for from langdetect library
DetectorFactory.seed = 0

# Metrics
prediction_count = Counter('predictions_total', 'Number of predictions made', ['category'])
correct_predictions = Counter('correct_predictions_total', 'Number of correct predictions', ['category'])
incorrect_predictions = Counter('incorrect_predictions_total', 'Number of incorrect predictions', ['category'])
prediction_confidence = Summary('prediction_confidence', 'Prediction confidence values')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')


@app.route('/')
def home():
    return "Welcome to the Issue Prediction API! Use the /predict endpoint to make predictions."


@app.route('/api/predict', methods=['POST'])
def predict_issue():
    try:
        data = request.get_json()

        # Extracting the 'title' and 'body' from the request
        title = data.get('title', '')
        body = data.get('body', '')

        # Ensure both title and body are strings and handle missing values
        text = str(title).strip() + ' ' + str(body).strip()
        print(f"input text: {text}")

        # Detect the language of the input
        language = detect(text)
        if language != 'en':
            return jsonify({
                "error": "The input language is not English. Please provide text in English.",
                "detected_language": language
            }), 400

        # Preprocess the input text (use your preprocess_text function)
        preprocessed_data = preprocess_text(text)
        # Debugging tokens
        tokens = preprocessed_data
        print(f"Tokens during inference: {tokens}")

        # Extract the preprocess text as a string
        preprocessed_text = ' '.join(tokens)

        # Predict using the model
        prediction = model.predict([preprocessed_text])
        print(f"Prediction type: {type(prediction)}")
        print(f"Prediction value: {prediction}")

        # Prediction probabilities
        y_probs = model.predict_proba([preprocessed_text])
        print("Prediction probabilities:", y_probs)

        # Convert prediction to a Python scalar (string or int)
        predicted_label = str(prediction[0])
        confidence = max(y_probs[0])  # Maximum probability as confidence score

        # Update metrics
        prediction_count.labels(predicted_label).inc()
        prediction_confidence.observe(confidence)

        # Generate a unique ID for the prediction
        issue_id = str(uuid.uuid4())

        # Extract important features from the input text using TF-IDF
        important_features = get_important_features_from_text(preprocessed_text)

        # Store the prediction in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        # noinspection SqlDialectInspection,SqlNoDataSourceInspection
        cursor.execute('''
                    INSERT INTO predictions (id, title, body, predicted_label, confidence) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (issue_id, title, body, predicted_label, confidence))
        conn.commit()
        conn.close()

        # LIME Explanation
        class_names = model.named_steps['classifier'].classes_.tolist()
        explainer = LimeTextExplainer(class_names=class_names)
        lime_exp = explainer.explain_instance(
            preprocessed_text,
            model.predict_proba,
            num_features=10
        )
        lime_explanation = lime_exp.as_list()  # List of (word, weight) tuples

        # Return the prediction and issue ID
        return jsonify({"id": issue_id, "predicted_label": predicted_label, "confidence": confidence, "important_features": important_features, "lime_explanation": lime_explanation}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/correct', methods=['POST'])
def correct_prediction():
    data = request.get_json()
    issue_id = data.get('id')
    corrected_label = data.get('corrected_label')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Retrieve existing prediction
        # noinspection SqlDialectInspection,SqlNoDataSourceInspection
        cursor.execute('SELECT predicted_label FROM predictions WHERE id = ?', (issue_id,))
        result = cursor.fetchone()

        if not result:
            print(f"Prediction ID not found")
            print(f"Received issue ID: {issue_id}")
            return jsonify({'error': 'Prediction ID not found'}), 404

        predicted_label = result[0]

        # Check if the corrected label matches the predicted label
        is_correct = (predicted_label == corrected_label)
        if is_correct:
            prediction_correct = "Yes"
            # Update metrics
            correct_predictions.labels(corrected_label).inc()
        else:
            prediction_correct = "No"
            # Update metrics
            incorrect_predictions.labels(corrected_label).inc()

        # calculate the accuracy metric based on the counters
        total_correct = sum(
            sample.value for metric in correct_predictions.collect() for sample in metric.samples
        )
        total_incorrect = sum(
            sample.value for metric in incorrect_predictions.collect() for sample in metric.samples
        )
        total_predictions = total_correct + total_incorrect
        if total_predictions > 0:
            accuracy = total_correct / total_predictions
            model_accuracy.set(accuracy)

        # Get the current timestamp
        timestamp = datetime.datetime.now()

        # Store correction in the corrections table
        print(f"Inserting correction for issue_id: {issue_id}, corrected_label: {corrected_label}")
        # noinspection SqlDialectInspection,SqlNoDataSourceInspection
        cursor.execute('''
                    UPDATE predictions 
                    SET corrected_label = ?, is_correct = ?, timestamp = ? 
                    WHERE id = ?
                ''', (corrected_label, prediction_correct, timestamp, issue_id))
        conn.commit()
        conn.close()

        # Return the response
        return jsonify({
            'id': issue_id,
            'corrected_label': corrected_label,
            'is_correct': is_correct,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/view_predictions', methods=['GET'])
def view_predictions():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Retrieve all rows from the predictions table
        # noinspection SqlDialectInspection,SqlNoDataSourceInspection
        cursor.execute('SELECT * FROM predictions')
        rows = cursor.fetchall()

        # Get the column names
        column_names = [description[0] for description in cursor.description]

        # Convert rows into a list of dictionaries and decode any bytes
        result = [
            {key: (value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value)
             for key, value in zip(column_names, row)}
            for row in rows
        ]

        conn.close()

        # Return the result as JSON
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(REGISTRY), content_type='text/plain')


@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    try:
        data = request.get_json()

        title = data.get('title', '')
        body = data.get('body', '')
        full_text = str(title).strip() + ' ' + str(body).strip()

        # Check language
        language = detect(full_text)
        if language != 'en':
            return jsonify({
                "error": "The input language is not English.",
                "detected_language": language
            }), 400

        # Preprocess
        preprocessed_tokens = preprocess_text(full_text)
        preprocessed_text = ' '.join(preprocessed_tokens)

        # LIME explainer for text
        class_names = model.named_steps['classifier'].classes_.tolist()
        explainer = LimeTextExplainer(class_names=class_names)

        # Explain instance
        explanation = explainer.explain_instance(
            preprocessed_text,
            model.predict_proba,
            num_features=10
        )

        # Get explanation as list
        explanation_data = explanation.as_list()

        return jsonify({
            "input_text": full_text,
            "predicted_label": model.predict([preprocessed_text])[0],
            "explanation": explanation_data  # List of tuples (word, weight)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
