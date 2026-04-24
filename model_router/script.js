document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const btnLoader = document.getElementById('btn-loader');
    const resultsSection = document.getElementById('results-section');
    const detectedModalitySpan = document.getElementById('detected-modality');
    const modelsContainer = document.getElementById('models-container');

    let currentType = 'text'; // default

    // Tab switching logic
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab
            tab.classList.add('active');
            currentType = tab.getAttribute('data-type');
            
            // Show corresponding content
            document.getElementById(`content-${currentType}`).classList.add('active');
            
            // Hide previous results
            resultsSection.classList.add('hidden');
        });
    });

    // File selection logic
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            fileNameDisplay.style.color = '#f8fafc'; // White text to indicate selection
        } else {
            fileNameDisplay.textContent = 'Choose an audio or video file';
            fileNameDisplay.style.color = '#94a3b8'; // Grey text
        }
    });

    // Submit logic
    submitBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('type', currentType);

        if (currentType === 'text') {
            const textVal = document.getElementById('text-input').value.trim();
            if (!textVal) {
                alert('Please enter a text query.');
                return;
            }
            formData.append('text', textVal);
        } else if (currentType === 'file') {
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file.');
                return;
            }
            formData.append('file', file);
        }

        // Show UI loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.7';
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                alert(`Error: ${data.error}`);
            } else {
                renderResults(data.modality, data.results);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the backend logic. Make sure app.py is running (python app.py) on localhost:5000.');
        } finally {
            // Restore UI state
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
            submitBtn.style.opacity = '1';
        }
    });

    function renderResults(modality, resultsObj) {
        detectedModalitySpan.textContent = modality;
        modelsContainer.innerHTML = ''; // clear previous

        // create cards dynamically
        Object.entries(resultsObj).forEach(([modelName, message]) => {
            const card = document.createElement('div');
            card.className = 'model-card';

            const modelInitial = modelName.split(' ')[1]; // "Model 1" -> "1"

            card.innerHTML = `
                <div class="model-icon">${modelInitial}</div>
                <div class="model-content">
                    <h3>${modelName}</h3>
                    <p>${message}</p>
                </div>
            `;
            modelsContainer.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
    }
});
