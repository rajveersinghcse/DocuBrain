document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const resultsCard = document.getElementById('resultsCard');
    const queryDisplay = document.getElementById('queryDisplay');
    const answerDisplay = document.getElementById('answerDisplay');
    const fileTypeDisplay = document.getElementById('fileTypeDisplay');
    const fileDetailsDisplay = document.getElementById('fileDetailsDisplay');
    const errorMessage = document.getElementById('errorMessage');
    
    // Initialize Bootstrap modals
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
    
    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(uploadForm);
        const file = formData.get('file');
        const query = formData.get('query');
        
        // Validate file and query
        if (!file || !query) {
            showError('Please select a file and enter a query.');
            return;
        }
        
        // Show loading modal
        loadingModal.show();
        
        try {
            // Upload file and process query
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            // Hide loading modal
            loadingModal.hide();
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to process file.');
            }
            
            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            loadingModal.hide();
            showError(error.message || 'An error occurred during processing.');
        }
    });
    
    // Display results
    function displayResults(data) {
        // Display query
        queryDisplay.textContent = data.query || '';
        
        // Display answer
        answerDisplay.textContent = data.answer || '';
        
        // Display file type
        const fileType = data.file_type_human || data.file_type || 'unknown';
        fileTypeDisplay.innerHTML = `<span class="file-badge badge-${fileType}">${fileType.toUpperCase()}</span>`;
        
        // Display additional details based on file type
        let detailsHTML = '';
        
        if (fileType === 'text' || fileType === 'pdf') {
            detailsHTML += `<p><strong>Sample:</strong> ${data.text_sample || 'N/A'}</p>`;
        } else if (fileType === 'image') {
            detailsHTML += `<p><strong>Summary:</strong> ${data.summary || 'N/A'}</p>`;
        } else if (fileType === 'video') {
            detailsHTML += `<p><strong>Visual Summary:</strong> ${data.visual_summary || 'N/A'}</p>`;
            detailsHTML += `<p><strong>Transcript:</strong> ${data.transcript || 'N/A'}</p>`;
        } else if (fileType === 'audio') {
            detailsHTML += `<p><strong>Transcript:</strong> ${data.transcript || 'N/A'}</p>`;
        } else if (fileType === 'tabular') {
            detailsHTML += `<p><strong>File Extension:</strong> ${data.file_extension || 'N/A'}</p>`;
            detailsHTML += `<p><strong>Columns:</strong> ${(data.columns || []).join(', ') || 'N/A'}</p>`;
            detailsHTML += `<p><strong>Row Count:</strong> ${data.row_count || 'N/A'}</p>`;
        }
        
        fileDetailsDisplay.innerHTML = detailsHTML;
        
        // Show results card
        resultsCard.classList.remove('d-none');
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Show error modal
    function showError(message) {
        errorMessage.textContent = message;
        errorModal.show();
    }
    
    // Clean up
    window.addEventListener('beforeunload', async () => {
        try {
            await fetch('/clean');
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    });
});