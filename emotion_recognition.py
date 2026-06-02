import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ======================================
# Configuration
# ======================================

DATASET_PATH = "dataset"
MODEL_PATH = "emotion_model.pkl"
RESULTS_PATH = "results.txt"
RANDOM_STATE = 42

# ======================================
# Emotion Labels
# ======================================

emotions = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ======================================
# Load Dataset
# ======================================

X = []
y = []

print("Loading audio files...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        emotion_code = file.split("-")[2]

        if emotion_code not in emotions:
            continue

        file_path = os.path.join(actor_path, file)

        try:

            audio, sample_rate = librosa.load(
                file_path,
                duration=3,
                offset=0.5
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            feature = np.mean(
                mfcc.T,
                axis=0
            )

            X.append(feature)
            y.append(emotions[emotion_code])

        except Exception:
            pass

print("Dataset Loaded Successfully")
print(f"Samples: {len(X)}")

# ======================================
# Prepare Data
# ======================================

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# ======================================
# Train Model
# ======================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train
)

# ======================================
# Predictions
# ======================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# ======================================
# Results
# ======================================

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
report = classification_report(
    y_test,
    predictions
)

print(report)

# ======================================
# Save Results
# ======================================

with open(RESULTS_PATH, "w") as file:

    file.write(
        f"Accuracy: {accuracy:.2%}\n\n"
    )

    file.write(
        "Classification Report\n\n"
    )

    file.write(report)

print("\nResults saved successfully!")

# ======================================
# Save Model
# ======================================

joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved successfully!")

# ======================================
# Confusion Matrix
# ======================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Emotion Recognition Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

print("Confusion matrix saved!")

# ======================================
# Emotion Distribution
# ======================================

emotion_counts = pd.Series(y).value_counts()

plt.figure(figsize=(8, 5))

emotion_counts.plot(
    kind="bar"
)

plt.title(
    "Emotion Distribution"
)

plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "emotion_distribution.png"
)

plt.close()

print("Emotion distribution graph saved!")

# ======================================
# Finished
# ======================================

print("\nProject executed successfully!")