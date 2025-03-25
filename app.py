from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load your trained model
try:
    model = load_model(r"D:\PROJECT\train_model\Multimodel_emotion_recognition_model_lstm.h5")
    print("Model loaded successfully!")
    model.summary()  # Print model summary to verify architecture
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Define label mapping
labels_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'pleasant surprised': 5, 'sad': 6}

# Define dataset path
TESS_PATH = r"C:\Users\ggani\Downloads\Cross-Attention Transformers for Multimodal Emotion Understanding in Human-Robot Interaction\multimodal emotion recognition in human robots interaction\TESS Toronto emotional speech set data"

# Path to save/load words_set
WORDS_SET_PATH = "words_set.pkl"

# Function to extract words from the dataset
def extract_words_from_dataset(dataset_path):
    words_set = set()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"TESS dataset directory not found at {dataset_path}")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("_")
                if len(parts) > 1:  # Ensure the filename has the expected format
                    word = parts[1].lower()  # Extract spoken word and convert to lowercase
                    words_set.add(word)
    
    if not words_set:
        raise ValueError("No words extracted from the dataset. Check the dataset path and file naming convention.")
    
    return words_set

# Load or extract words_set
if os.path.exists(WORDS_SET_PATH):
    # Load words_set from file if it exists
    with open(WORDS_SET_PATH, 'rb') as f:
        words_set = pickle.load(f)
    print(f"Loaded words_set from {WORDS_SET_PATH}")
else:
    # Extract words from the dataset and save to file
    words_set = extract_words_from_dataset(TESS_PATH)
    with open(WORDS_SET_PATH, 'wb') as f:
        pickle.dump(words_set, f)
    print(f"Saved words_set to {WORDS_SET_PATH}")

# Initialize word encoder with all words from the dataset
word_encoder = LabelEncoder()
word_encoder.fit(list(words_set))
print(f"Number of unique words loaded: {len(words_set)}")
print(f"Words (first 10): {list(words_set)[:10]}")  # Print first 10 words for verification

def extract_features(audio_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)
    return mfcc

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files or 'word' not in request.form:
        return jsonify({'error': 'Missing audio file or spoken word'}), 400
    
    audio_file = request.files['audio']
    spoken_word = request.form['word'].lower()  # Convert to lowercase for consistency
    
    # Validate audio file extension
    if not audio_file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files are supported'}), 400
    
    # Save uploaded audio temporarily
    audio_path = os.path.join('uploads', audio_file.filename)
    os.makedirs('uploads', exist_ok=True)
    audio_file.save(audio_path)
    
    try:
        # Process audio
        audio_features = extract_features(audio_path)
        audio_features = np.expand_dims(audio_features, axis=0)
        print("Audio Features Shape:", audio_features.shape)
        
        # Process text
        if spoken_word in word_encoder.classes_:
            text_features = word_encoder.transform([spoken_word])
            text_features = np.expand_dims(text_features, axis=-1)
            print("Text Features Shape:", text_features.shape)
        else:
            os.remove(audio_path)
            return jsonify({'error': f'Word "{spoken_word}" not in vocabulary. Available words: {list(word_encoder.classes_)[:10]}...'}), 400
        
        # Predict emotion
        prediction = model.predict([audio_features, text_features])
        print("Prediction Shape:", prediction.shape)
        print("Prediction Probabilities:", prediction)
        predicted_label = list(labels_map.keys())[np.argmax(prediction)]
        print("Predicted Label:", predicted_label)
        
        # Clean up
        os.remove(audio_path)
        
        return jsonify({'emotion': predicted_label})
    
    except Exception as e:
        os.remove(audio_path)
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)