const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadedImage = document.getElementById('uploadedImage');

// Click to open file dialog
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Show the uploaded image
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            uploadedImage.src = event.target.result;
            uploadedImage.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', (e) => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;  // Set dropped file to input
        const reader = new FileReader();
        reader.onload = function (event) {
            uploadedImage.src = event.target.result;
            uploadedImage.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});
