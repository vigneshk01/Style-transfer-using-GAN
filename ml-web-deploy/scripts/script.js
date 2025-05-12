function handleSingleImageUpload(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const fileInput = form.querySelector('input[type="file"]');
    console.log('handleSingleImageUpload')

    if (!fileInput || fileInput.files.length === 0) {
        console.error("No file selected.");
        return;
    }

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.blob()) // Check if the response is ok
    .then(blob => blob.text() )
    .then(text => {
        const obj = JSON.parse(text)
        const filename = obj.filename
        return filename})
    .then(filename => { 
        const imageUrl = '/static/static/uploads/' + filename;
        const displayedImage = document.getElementById('converted-image');
        if (displayedImage) {
            displayedImage.src = imageUrl;
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

    fetch('/batch', {
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
