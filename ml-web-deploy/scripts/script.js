function handleSingleImageUpload(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const fileInput = form.querySelector('input[type="file"]');

    if (!fileInput || fileInput.files.length === 0) {
        console.error("No file selected.");
        return;
    }

    fetch('/upload-file', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())  // Expect JSON response with filename
    .then(data => {
        if (data.filename) {
            const imageUrl = `/static/static/uploads/${data.filename}`;  // Build the image URL
            const displayedImage = document.getElementById('converted-image');
            if (displayedImage) {
                displayedImage.src = imageUrl;  // Set the image source to the URL
                displayedImage.alt = "Transformed Image";  // Add alt text for accessibility
            }
        } else {
            console.error("No filename returned by the server.");
        }
    })
    .catch(error => {
        console.error('Error during single image upload:', error);
    });
}

function handleBatchUpload(event) {
    event.preventDefault();

    const form = event.target;
    const fileInput = form.querySelector('input[type="file"]');

    // Check if there are more than 10 files selected
    if (fileInput.files.length > 10) {
        alert("You can only upload a maximum of 10 images.");
        return;
    }

    const formData = new FormData(form);

    fetch('/batch-upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Batch conversion completed. Files saved to: " + data.export_path);
        } else {
            alert("Batch conversion failed: " + data.error);
        }
    })
    .catch(error => {
        console.error('Error during batch upload:', error);
    });
}

// Attach handlers
document.addEventListener('DOMContentLoaded', () => {
    const singleForm = document.getElementById('single-upload-form');
    const batchForm = document.getElementById('batch-upload-form');

    if (singleForm) {
        singleForm.addEventListener('submit', handleSingleImageUpload);
    }
    if (batchForm) {
        batchForm.addEventListener('submit', handleBatchUpload);
    }
});
