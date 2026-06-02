# Speech Emotion Recognition Using Machine Learning

## Overview

This project is a Speech Emotion Recognition (SER) system developed using Machine Learning and the RAVDESS Emotional Speech Dataset. The model analyzes speech audio files and predicts the emotional state expressed by the speaker.

The objective of this project is to automatically recognize human emotions from voice recordings using audio feature extraction and classification techniques.

---

## Dataset

### RAVDESS Emotional Speech Audio Dataset

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is a widely used dataset for emotion recognition research.

### Dataset Details

- Total Audio Samples Used: **1248**
- Audio Format: **WAV**
- Sampling Rate: **48 kHz**
- Actors: **24 Professional Actors**
- Language: **English**

### Emotions Detected

| Emotion Code | Emotion |
|-------------|----------|
| 01 | Neutral |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## Features Extracted

The model uses **MFCC (Mel Frequency Cepstral Coefficients)** extracted from audio signals.

MFCC features are widely used in speech processing because they effectively represent the characteristics of human speech.

### Feature Extraction Method

- Audio Loading using Librosa
- Noise Reduction through audio trimming
- MFCC Extraction
- Feature Averaging
- Feature Vector Creation

---

## Machine Learning Algorithm

### Random Forest Classifier

The Random Forest algorithm was selected because:

- High classification performance
- Handles multi-class classification effectively
- Less prone to overfitting
- Works well with extracted MFCC features

### Model Parameters

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

---

## Project Workflow

### 1. Data Collection

Load all audio files from the RAVDESS dataset.

### 2. Emotion Label Extraction

Extract emotion labels from the audio file names.

### 3. Audio Feature Extraction

Extract MFCC features using Librosa.

### 4. Dataset Preparation

Convert extracted features into training data.

### 5. Train-Test Split

Split dataset into:

- Training Set: 80%
- Testing Set: 20%

### 6. Model Training

Train Random Forest Classifier using extracted MFCC features.

### 7. Model Evaluation

Evaluate model using:

- Accuracy Score
- Classification Report
- Confusion Matrix

### 8. Model Saving

Save trained model using Joblib.

---

## Model Performance

### Overall Accuracy

**62.80%**

### Classification Report

| Emotion | Precision | Recall | F1-Score |
|----------|-----------|---------|-----------|
| Angry | 0.71 | 0.74 | 0.72 |
| Disgust | 0.56 | 0.57 | 0.56 |
| Fearful | 0.52 | 0.57 | 0.54 |
| Happy | 0.71 | 0.53 | 0.61 |
| Neutral | 0.62 | 0.65 | 0.64 |
| Sad | 0.57 | 0.50 | 0.53 |
| Surprised | 0.68 | 0.84 | 0.75 |

---

## Confusion Matrix

The confusion matrix provides a detailed breakdown of model predictions across all emotion classes.

![Confusion Matrix](confusion_matrix.png)

---

## Emotion Distribution

The following graph shows the distribution of emotion samples present in the dataset.

![Emotion Distribution](emotion_distribution.png)

---

## Generated Files

| File | Description |
|--------|-------------|
| emotion_recognition.py | Main project source code |
| emotion_model.pkl | Trained Random Forest model |
| confusion_matrix.png | Confusion matrix visualization |
| emotion_distribution.png | Dataset emotion distribution graph |
| results.txt | Model evaluation results |
| README.md | Project documentation |
| requirements.txt | Required Python libraries |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/EmotionRecognition.git
```

### Navigate to Project Folder

```bash
cd EmotionRecognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the following command:

```bash
python emotion_recognition.py
```

### Expected Output

```text
Dataset Loaded Successfully
Samples: 1248

Accuracy: 62.80%

Results saved successfully!
Model saved successfully!
Confusion matrix saved!
Emotion distribution graph saved!

Project executed successfully!
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- Scikit-Learn
- Joblib
- Matplotlib
- Seaborn

---

## Future Improvements

- Deep Learning using CNNs
- LSTM-based Speech Emotion Recognition
- Real-Time Emotion Prediction
- Speech-to-Text Integration
- Web Application Deployment using Flask

---

## Author

### Anubhab Pal

Machine Learning Intern — CodeAlpha

---

## License

This project is developed for educational and internship purposes under the CodeAlpha Machine Learning Internship Program.
