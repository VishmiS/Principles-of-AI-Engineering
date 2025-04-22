import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model import create_model, train_model, save_model, predict_category, extract_important_features

# Load preprocessed data
df = pd.read_excel(r'C:\ws2024-principles-of-ai-engineering\preprocessed.xlsx')

# Ensure clean_text and issue_label columns are present
if 'text' not in df.columns or 'issue_label' not in df.columns:
    raise ValueError("The preprocessed data must have 'clean_text' and 'issue_label' columns.")

# Feature and target
X = df['text']
y = df['issue_label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = create_model()

# Train the model
train_model(model, X_train, y_train)
print("Model trained successfully!")

# Save the trained model
model_path = r'C:\ws2024-principles-of-ai-engineering\random_forest_model.pkl'
save_model(model, model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Extract feature importances
tfidf_step = model.named_steps['tfidf']  # Access the TF-IDF step in the pipeline
classifier_step = model.named_steps['classifier']  # Access the classifier step in the pipeline

if hasattr(classifier_step, "feature_importances_"):
    importances = classifier_step.feature_importances_
    top_features = extract_important_features(tfidf_step, importances, n=10)
    print("Top 10 important features:")
    for feature in top_features:
        print(f"{feature['feature_name']}: {feature['importance_score']}")
else:
    print("The classifier does not provide feature importances.")

# Predict categories for new inputs
random_inputs = [
    "The application crashes when I click the submit button on the form",
    "It would be great if we could add a dark mode option to the settings",
    "Can someone explain how to use the API with Python?"
]

predicted_categories = predict_category(random_inputs, model)

for text, prediction in zip(random_inputs, predicted_categories):
    print(f"Input: {text}\nPredicted Category: {prediction}\n")
