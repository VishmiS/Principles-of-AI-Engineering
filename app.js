const apiUrl = "http://127.0.0.1:5000/api"; // Ensure this is the correct backend URL

document.addEventListener("DOMContentLoaded", () => {
    // Attach event listeners after DOM is fully loaded
    document.getElementById("predict-btn").addEventListener("click", predictIssue);
    document.getElementById("submit-btn").addEventListener("click", submitCorrection);
    document.getElementById("reset-btn").addEventListener("click", resetForm); // Reset button event listener
});

async function predictIssue() {
    const title = document.getElementById('title').value.trim();
    const body = document.getElementById('body').value.trim();

    if (!title || !body) {
        alert("Please enter both title and body.");
        return;
    }

    const data = { title, body };

    try {
        const response = await fetch(`${apiUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            if (result.error && result.detected_language) {
                displayErrorMessage(`Error: ${result.error} Detected language: ${result.detected_language}`);
            } else {
                displayErrorMessage(`Error: ${result.error || 'Unexpected error occurred'}`);
            }

            throw new Error(`Server returned status ${response.status}`);
        }

        console.log('Prediction Result:', result);

        // Check if required fields exist in the result
        if (result && result.predicted_label && result.id) {
            document.getElementById('predicted-label').innerText = result.predicted_label;
            document.getElementById('prediction-id').innerText = result.id;
            document.getElementById('confidence-score').innerText = result.confidence.toFixed(2);

            // Check confidence level and display warning if necessary
            if (result.confidence < 0.5) {
                document.getElementById("confidence-warning").innerHTML =
                    "Warning: Low confidence in the prediction. Consider reviewing it.";
                document.getElementById("confidence-warning").style.display = "block";
            } else {
                document.getElementById("confidence-warning").style.display = "none";
            }

            // Highlight input text with important features
            highlightInputText(title, body, result.important_features);

            document.getElementById('prediction-result').style.display = 'block';
            document.getElementById('predict-btn').disabled = true; // Disable predict button after prediction
        } else {
            alert('Unexpected response format');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
    }
}

function displayErrorMessage(message) {
    const errorContainer = document.getElementById('error-message');
    if (errorContainer) {
        errorContainer.innerText = message;
        errorContainer.style.display = 'block';
    }
}

function highlightInputText(title, body, importantFeatures = []) {
    let inputText = `${title}\n\n${body}`;
    importantFeatures.sort((a, b) => b.importance_score - a.importance_score);

    importantFeatures.forEach(feature => {
        const importanceScore = feature.importance_score;
        const color = importanceScore > 0.7
            ? 'red'
            : importanceScore > 0.4
            ? 'orange'
            : 'yellow';

        const regex = new RegExp(`\\b(${feature.feature_name})\\b`, 'gi');
        inputText = inputText.replace(regex, `<span style="background-color: ${color}; font-weight: bold;">$1</span>`);
    });

    document.getElementById('highlighted-input').innerHTML = inputText;
    document.getElementById('highlighted-text').style.display = 'block';
}

async function submitCorrection() {
    const issueId = document.getElementById('prediction-id').innerText;
    const correctedLabel = document.querySelector('input[name="label"]:checked');

    if (!correctedLabel) {
        alert("Please select the corrected label.");
        return;
    }

    const data = { id: issueId, corrected_label: correctedLabel.value };

    try {
        const response = await fetch(`${apiUrl}/correct`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const result = await response.json();
        console.log('Correction Result:', result);

        alert('Correction submitted successfully.');
        // document.getElementById('prediction-result').style.display = 'none'; // Hide result section
        // document.getElementById('predict-btn').disabled = false; // Re-enable predict button
        // resetForm(); // Reset the form after submitting correction
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
    }
}

// Function to reset the form and prediction result section
function resetForm() {
    document.getElementById('prediction-form').reset(); // Reset form fields
    document.getElementById('prediction-result').style.display = 'none'; // Hide prediction result section
    document.getElementById('predict-btn').disabled = false; // Re-enable predict button
    document.getElementById('error-message').style.display = 'none'; // Hide confidence warning
    document.getElementById('confidence-warning').style.display = 'none'; // Hide confidence warning
    document.getElementById('important-features').innerHTML = ''; // Clear important features list
// Reset the radio buttons
const radioButtons = document.querySelectorAll('input[name="label"]');
radioButtons.forEach(radio => radio.checked = false); // Uncheck all radio buttons
}
