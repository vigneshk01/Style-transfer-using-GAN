// This script handles the form submission for file upload and displays the converted image.
function handleFormSubmit(event) {
    event.preventDefault();  
    const formData = new FormData(event.target);  
    const fileInput = document.getElementById('file-input');
    if (fileInput && fileInput.files.length > 0) {
        formData.append('image', fileInput.files[0]);  
    } else {
        console.error("Element 'file-input' not found or no file selected.");
    }
    fetch('/', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                console.error('Server error response:', errorData);
                throw new Error(`HTTP error! status: ${response.status}`);
            }).catch(() => {
                throw new Error(`HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data && data.filename) {
            console.log('File uploaded successfully:', data.filename);
            const displayedImage = document.getElementById('converted-image');
            if (displayedImage && data.filename) {
                displayedImage.src = `/static/static/uploads/converted_${data.filename}`;
            }
        } else if (data && data.error) {
            console.error('Server error:', data.error);
        } else {
            console.error('Unexpected server response:', data);
        }
    })
    .catch(error => {
        console.error('Error during file upload:', error);
    });
}

// Attach the form submit handler
const uploadForm = document.getElementById('upload-form');
if (uploadForm) {
    uploadForm.addEventListener('submit', handleFormSubmit);
} else {
    console.error("Element 'upload-form' not found.");
}