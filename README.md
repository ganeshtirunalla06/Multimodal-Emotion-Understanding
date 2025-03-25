# Cross-Attention Transformers for Multimodal Emotion Understanding in Human-Robot Interaction

## Overview  
Emotion recognition in human-robot interaction is a rapidly growing research area, contributing to advancements in **artificial intelligence**, **human-computer interaction**, and **affective computing**. This project focuses on developing a **multimodal emotion recognition system** using **Cross-Attention Transformers**. The model processes **audio and text inputs** to accurately classify emotions, improving interactions between humans and robots.

The project is built as a **Flask web application**, allowing users to upload an **audio file (.wav format)** and input a **spoken word** corresponding to the audio. The system analyzes these inputs and classifies the emotion using **deep learning models**. The ultimate goal is to **enhance emotional intelligence in robots**, enabling them to respond appropriately in social scenarios.

## Features  
- **Multimodal Emotion Recognition**: Combines **speech and text inputs** for improved accuracy.
- **Deep Learning-Based Approach**: Utilizes **Cross-Attention Transformers** and neural networks to extract and fuse features.
- **Flask Web Interface**: Provides a simple, interactive UI for users to analyze emotions.
- **Real-Time Predictions**: Classifies emotions instantly after uploading audio and text.
- **Audio Feature Extraction**: Employs **MFCC (Mel-Frequency Cepstral Coefficients)** for speech analysis.
- **Cross-Attention Mechanism**: Enhances performance by correlating information between text and audio modalities.
- **Interactive UI**: Built with **Bootstrap**, **JavaScript**, and custom CSS for a seamless user experience.

## Technology Stack  
### Back-End  
- **Python**: Core programming language (version 3.8+).
- **Flask**: Web framework for handling requests and serving web pages.
- **TensorFlow/Keras**: Deep learning framework for model training and inference.
- **Librosa**: Library for extracting audio features (e.g., MFCCs).
- **Scikit-learn**: Used for data preprocessing and evaluation.
- **NumPy/Pandas**: Libraries for data manipulation and analysis.

### Front-End  
- **HTML/CSS**: For structuring and styling the user interface.
- **Bootstrap**: Ensures a responsive and modern design.
- **JavaScript**: Handles client-side interactions and form submissions.
- **Boxicons**: Provides modern icons for social media links.

### Deployment  
- **Flask Web Server**: Runs locally for testing and evaluation.
- **Google Colab/Jupyter Notebook**: Used for model training and evaluation.

## Installation  

### Prerequisites  
Ensure you have **Python 3.10+** installed on your system. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

1. **Run the Flask server:**  
   ```bash
   python app.py
   ```
2. **Open your browser** and navigate to:  
   ```
   http://127.0.0.1:5000
   ```

## Usage  
### **Step-by-Step Guide**  
1. **Upload an audio file (.wav format).**  
2. **Enter the spoken word corresponding to the audio file.**  
3. **Click "Analyze Emotion"** to get the predicted emotion.  
4. The system will **display the detected emotion** on the UI.  

### **Example Use Cases**  
- **Human-Robot Interaction**: Helps robots understand and respond to human emotions appropriately.
- **Mental Health Analysis**: Can be adapted for detecting emotions in speech therapy.
- **Customer Support**: Enhances chatbot interactions by recognizing user emotions.
- **Entertainment Industry**: Helps in analyzing audience reactions in media applications.

## Model Details  
### **How It Works**  
1. **Audio Processing:**  
   - Extracts **MFCC features** from the input audio.
   - Converts them into numerical representations suitable for deep learning.
2. **Text Processing:**  
   - Encodes the **spoken word** into a numerical format.
   - Uses a **pretrained embedding layer** for feature extraction.
3. **Fusion Using Cross-Attention Transformers:**  
   - The model **aligns and correlates** features from both **audio and text**.
   - Uses **attention mechanisms** to capture relevant information from both inputs.
4. **Emotion Prediction:**  
   - The final output layer classifies the input into **one of seven emotions**:
     - **Angry**
     - **Happy**
     - **Sad**
     - **Neutral**
     - **Fear**
     - **Disgust**
     - **Pleasant Surprise**

### **Training Dataset**  
- **Dataset Used:** TESS Toronto Emotional Speech Set.
- **Data Augmentation:** Used to enhance training robustness.
- **Split Ratio:** 80% Training, 20% Testing.

### **Performance Metrics**  
- **Accuracy:** ~98% on test data.
- **Confusion Matrix** used to evaluate misclassification.
- **Precision, Recall, and F1-Score** calculated for each class.

## Folder Structure  
```
📂 emotion-recognition  
│── 📂 static/           # CSS & JS files  
│── 📂 templates/        # HTML templates  
│── 📂 uploads/          # Uploaded audio files  
│── app.py             # Flask backend  
│── index.html         # Web UI  
│── style.css          # CSS styles  
│── requirements.txt    # Python dependencies  
```

## Future Enhancements  
1. **Support for More Languages:** Extend emotion recognition to **multiple languages**.
2. **Real-Time Processing:** Implement **streaming support** for live audio emotion detection.
3. **Facial Emotion Recognition:** Integrate **video-based emotion recognition**.
4. **Edge Deployment:** Optimize the model for deployment on **mobile devices**.
5. **Self-Learning AI:** Implement a **self-improving model** that learns from user feedback.
6. **More Granular Emotions:** Enhance classification to detect **subtle emotional states**.
7. **Improved UI/UX:** Add **data visualization** for model interpretation.

## Contributors  
**Ganesh Kumar Tirunalla**  
- [LinkedIn](https://www.linkedin.com/in/ganesh-kumar-tirunalla-39609024a)  
- [GitHub](https://github.com/ganeshtirunalla06)  
- 📧 Email: [ganeshtirunalla06@gmail.com](mailto:ganeshtirunalla06@gmail.com)  

## License  
This project is licensed under the **MIT License**.

---  
⭐ **If you found this project helpful, consider giving it a star on GitHub!** 🚀
```
```

