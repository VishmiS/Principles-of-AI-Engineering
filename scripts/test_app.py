import pytest
from app import app
import nltk
from unittest.mock import patch, MagicMock, ANY
from db import get_db_connection


# Create a test app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client  # Provide the test client for use in test cases


# 1. Test NLTK Data Download
@patch('nltk.download')
def test_download_nltk_data(mock_nltk_download):
    mock_nltk_download.return_value = None  # Simulate successful download
    try:
        nltk.download('punkt')  # Function call under test
        mock_nltk_download.assert_called_with('punkt')  # Ensure NLTK's download was called
    except Exception as e:
        pytest.fail(f"Test failed due to exception: {e}")


# 2. Test Home Endpoint
def test_home(client):
    response = client.get('/')  # Simulate a GET request to the home endpoint
    assert response.status_code == 200
    assert b"Welcome to the Issue Prediction API!" in response.data  # Ensure the correct response message


# 3. Test Database Connection Initialization
def test_db_connection():
    conn = get_db_connection()  # Ensure the database connection is working
    assert conn is not None  # Ensure a valid connection object is returned
    conn.close()  # Close the connection after the test


# 4. Test Prediction Endpoint with Mock Data
@patch('app.get_db_connection')
def test_predict_issue_with_real_model(mock_get_db_connection, client):
    # Test input data
    mock_data = {
        "title": "Bug in login system",
        "body": "The login system crashes when password is too long."
    }

    # Mock database connection to prevent live DB access
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_db_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Call the endpoint with real model and preprocessing
    response = client.post('/api/predict', json=mock_data)
    assert response.status_code == 200  # Ensure the status code is 200
    json_data = response.get_json()

    # Assertions
    assert response.status_code == 200
    assert 'id' in json_data
    assert 'predicted_label' in json_data

    # Ensure the prediction is a non-empty string (your model output)
    assert isinstance(json_data['predicted_label'], str)

    # Verify database interaction was attempted
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()

    # Optionally, print to verify output manually (remove in production tests)
    print(json_data)


# 5. Test Prediction Correction Endpoint
@patch('app.get_db_connection')  # Mock the database connection function
def test_correct_prediction(mock_get_db_connection, client):
    # Mock input data
    mock_data = {
        "id": "91e897be-dcee-4da2-b828-0a65122da033",  # Mock prediction ID
        "corrected_label": "bug"  # Corrected label to test
    }

    # Set up mock database behavior
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_db_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Simulate the behavior of a SELECT query
    predicted_label = "enhancement"  # Simulate a different predicted label
    mock_cursor.fetchone.return_value = (predicted_label,)  # Return as tuple

    # Simulate the behavior of the INSERT query
    mock_cursor.execute.return_value = None  # Simulate successful query execution
    mock_conn.commit.return_value = None  # Mock commit to prevent real DB interaction

    # Log the expected behavior
    print(f"Mocked data: {mock_data}")
    print(f"Predicted label from mock: {predicted_label}")

    # Perform the POST request to the correct endpoint
    response = client.post('/api/correct', json=mock_data)

    # Debugging output: print response
    print(f"Response: {response.data}")

    # Assertions
    assert response.status_code == 200  # Ensure the status code is 200
    json_data = response.get_json()

    # Ensure response contains the expected keys and values
    assert 'id' in json_data
    assert 'corrected_label' in json_data
    assert 'is_correct' in json_data
    assert isinstance(json_data['corrected_label'], str)

    # Verify correct label comparison logic
    expected_correct_status = (mock_data['corrected_label'] == predicted_label)  # False
    assert json_data['id'] == mock_data['id']
    assert json_data['corrected_label'] == mock_data['corrected_label']
    assert json_data['is_correct'] == expected_correct_status

    # Debugging: Print the actual calls made
    print(mock_cursor.execute.call_args_list)

    print(f"Before executing: {mock_data}")
    # Assert that execute was called twice: once for SELECT and once for INSERT
    # noinspection SqlDialectInspection,SqlNoDataSourceInspection
    mock_cursor.execute.assert_any_call(
        'SELECT predicted_label FROM predictions WHERE id = ?',
        ('91e897be-dcee-4da2-b828-0a65122da033',)
    )
    # Assert the UPDATE query was executed as expected
    # noinspection SqlDialectInspection,SqlNoDataSourceInspection
    mock_cursor.execute.assert_any_call(
        '\n                    UPDATE predictions \n                    SET corrected_label = ?, is_correct = ?, timestamp = ? \n                    WHERE id = ?\n                ',
        ('bug', 'No', ANY, '91e897be-dcee-4da2-b828-0a65122da033')
    )

    # Verify commit was called
    mock_conn.commit.assert_called_once()

    # Optionally print to verify output manually (remove in production tests)
    print(json_data)
