const apiUrl = "http://127.0.0.1:5000/api"; // Ensure this is the correct backend URL

document.addEventListener("DOMContentLoaded", () => {
    // Attach event listeners after DOM is fully loaded
    document.getElementById("predict-btn").addEventListener("click", predictIssue);
    document.getElementById("submit-btn").addEventListener("click", submitCorrection);
    document.getElementById("reset-btn").addEventListener("click", resetForm); // Reset button event listener
    // Hide lime explanation on load
    document.getElementById('lime-explanation').style.display = 'none';
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

            // Render LIME explanation and make it visible
            if (result.lime_explanation) {
                renderLimeExplanation(result.lime_explanation);
            }

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
    console.log("Reset function triggered");
    document.getElementById('prediction-form').reset(); // Reset form fields
    document.getElementById('prediction-result').style.display = 'none'; // Hide prediction result section
    document.getElementById('predict-btn').disabled = false; // Re-enable predict button
    document.getElementById('error-message').style.display = 'none'; // Hide confidence warning
    document.getElementById('confidence-warning').style.display = 'none'; // Hide confidence warning
    document.getElementById('important-features').innerHTML = ''; // Clear important features list
    document.getElementById('lime-explanation').style.display = 'none';
    document.getElementById('lime-table-body').innerHTML = '';
// Reset the radio buttons
const radioButtons = document.querySelectorAll('input[name="label"]');
radioButtons.forEach(radio => radio.checked = false); // Uncheck all radio buttons
}

function renderLimeExplanation(explanation) {
    const limeSection = document.getElementById('lime-explanation');
    const outputDiv = document.getElementById('lime-table-body');
    outputDiv.innerHTML = ''; // Clear old content

    // Legend (better styled)
    const legend = document.createElement('div');
    legend.innerHTML = `
        <div style="margin-bottom: 16px; font-size: 14px;">
            <strong>Legend:</strong>
            <span style="background-color: rgba(0, 128, 0, 0.2); color: black; padding: 4px 8px; margin-left: 12px; border-radius: 6px;">Supports Prediction</span>
            <span style="background-color: rgba(255, 0, 0, 0.2); color: black; padding: 4px 8px; margin-left: 12px; border-radius: 6px;">Against Prediction</span>
        </div>
    `;
    outputDiv.appendChild(legend);

    // Line for each token
    const explanationWrapper = document.createElement('div');
    explanationWrapper.style.display = 'flex';
    explanationWrapper.style.flexWrap = 'wrap';
    explanationWrapper.style.gap = '12px';
    explanationWrapper.style.alignItems = 'flex-end';

    explanation.forEach(([word, weight]) => {
        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.alignItems = 'center';
        wrapper.style.minWidth = '50px';
        wrapper.style.wordBreak = 'break-word';

        const weightLabel = document.createElement('div');
        weightLabel.textContent = weight.toFixed(4);
        weightLabel.style.fontSize = '12px';
        weightLabel.style.marginBottom = '4px';
        weightLabel.style.color = weight >= 0 ? 'green' : 'red';

        const wordBox = document.createElement('div');
        wordBox.textContent = word;
        wordBox.style.padding = '6px 10px';
        wordBox.style.borderRadius = '8px';
        wordBox.style.backgroundColor = weight >= 0
            ? `rgba(0, 128, 0, ${Math.min(Math.abs(weight), 1)})`
            : `rgba(255, 0, 0, ${Math.min(Math.abs(weight), 1)})`;
        wordBox.style.color = '#000';
        wordBox.style.fontWeight = 'bold';
        wordBox.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
        wordBox.style.textAlign = 'center';

        wrapper.appendChild(weightLabel);
        wrapper.appendChild(wordBox);
        explanationWrapper.appendChild(wrapper);
    });

    outputDiv.appendChild(explanationWrapper);
    limeSection.style.display = 'block';
}
