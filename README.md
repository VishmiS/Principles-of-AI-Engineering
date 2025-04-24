# AI-Powered Issue Classification System  
### Real-World Simulation of an End-to-End AI Engineering Pipeline

---

## 📘 Project Overview

This project is a full-stack implementation of an AI-powered issue classification system. Designed for integration into software development workflows, the system automatically categorizes issues into:

- `enhancement`
- `bug`
- `question`

It incorporates data preprocessing, model training, REST API development with Flask, database storage, monitoring with Prometheus and Grafana, automated testing, and a lightweight frontend interface. The structure and tooling reflect best practices in AI engineering and DevOps.

---

#### Project Structure


├── datasets/                 # CSV and compressed data samples  
│   ├── predictions.csv  
│   ├── sample1.csv.gz  
│   └── sample2.csv.gz  
├── database/                 # SQLite database file  
│   └── predictions.db  
├── scripts/                  # Core application logic and scripts  
│   ├── app.py                # Main Flask application  
│   ├── db.py                 # Database access logic  
│   ├── model.py              # Model loading and prediction  
│   ├── preprocessing.py      # Text preprocessing pipeline  
│   ├── server.py             # Server runner script  
│   ├── test_app.py           # Pytest tests for API endpoints  
│   └── train.py              # Model training logic  
├── .gitignore                # Ensures large or unnecessary files are excluded  
├── .gitlab-ci.yml            # GitLab CI/CD pipeline configuration  
├── requirements.txt          # Python dependencies  
├── docker-compose.yml        # Container setup for Prometheus & Grafana  
├── prometheus.yml            # Prometheus scrape config  
├── index.html                # Frontend UI  
├── app.js                    # JavaScript logic for the client UI  
├── styles.css                # Styling for the frontend  
├── .coverage / coverage.xml  # Code coverage reports  
├── README.md                 # Project documentation

---

## Getting Started

### 1. Clone the Repository
`git clone https://github.com/VishmiS/Principles-of-AI-Engineering.git`  


### 2. Set Up Environment
Using venv:  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`

### 3. Run the Flask App
`python scripts/app.py`



## REST API Endpoints
- `POST /api/predict`
  - Accepts an issue's title and description. Returns a predicted label and generated issue ID, and logs the data to the database.
- `POST /api/correct`
  - Accepts a corrected label. Compares it to the previous prediction and updates the database for tracking performance.


## AI Model
- Trained using a Random Forest Classifier from scikit-learn
- Preprocessing includes lowercasing, punctuation removal, tokenization, stopword filtering, and lemmatization
- Modular architecture allows re-use of the preprocessing pipeline across training and inference

## Testing & CI/CD
- All endpoints are tested using pytest
- Test coverage tracked with pytest-cov
- GitLab CI/CD pipeline (.gitlab-ci.yml) runs tests and generates coverage reports on every push
- To run tests manually: `pytest scripts/test_app.py --cov=scripts --cov-report=xml`

## Monitoring & Metrics
Using Prometheus and Grafana to track:
- Number of predictions per category
- Accuracy and confidence over time
- Correct vs incorrect prediction distribution
- Launch with: `docker-compose up`
    - Prometheus: `http://localhost:9090`
    - Grafana: `http://localhost:3000`

## Frontend Client
The included client allows interactive issue submission and classification through a simple UI.
- Inputs are validated
- Model confidence is shown
- Feedback loop allows corrections

## Safety & Fairness Features
- Language Detection: Rejects or flags non-English input
- Confidence Scoring: Displayed to help users assess trust
- Data Visualization: Performed to inspect dataset balance and potential bias
