document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const predictBtn = document.getElementById('predictBtn');
    const emotionSpan = document.getElementById('emotion');

    // Show loading state
    predictBtn.disabled = true;
    predictBtn.classList.add('loading');
    predictBtn.textContent = 'Analyzing...';

    const formData = new FormData();
    formData.append('audio', document.getElementById('audio').files[0]);
    formData.append('word', document.getElementById('word').value);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            emotionSpan.textContent = data.emotion;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing the emotion.');
    } finally {
        // Reset button state
        predictBtn.disabled = false;
        predictBtn.classList.remove('loading');
        predictBtn.textContent = 'Analyze Emotion';
    }
});