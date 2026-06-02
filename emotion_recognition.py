
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

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

def load_dataset():

    print("Loading audio files...")

    X = []
    y = []

    for actor in os.listdir(DATASET_PATH):

        actor_path = os.path.join(
            DATASET_PATH,
            actor
        )

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            if not file.endswith(".wav"):
                continue

            emotion_code = file.split("-")[2]

            if emotion_code not in emotions:
                continue

            file_path = os.path.join(
                actor_path,
                file
            )

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
                y.append(
                    emotions[emotion_code]
                )

            except Exception:
                pass

    print("Dataset Loaded Successfully")
    print(f"Samples: {len(X)}")

    return np.array(X), y


# ======================================
# Train Model
# ======================================

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ======================================
# Save Results
# ======================================

def save_results(accuracy, report, y):

    with open(
        RESULTS_PATH,
        "w"
    ) as file:

        file.write(
            "Speech Emotion Recognition Results\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.2%}\n\n"
        )

        file.write(
            "Classification Report\n\n"
        )

        file.write(report)

    print(
        "\nResults saved successfully!"
    )


# ======================================
# Plot Confusion Matrix
# ======================================

def plot_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
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

    print(
        "Confusion matrix saved!"
    )


# ======================================
# Plot Emotion Distribution
# ======================================

def plot_distribution(y):

    emotion_counts = pd.Series(
        y
    ).value_counts()

    plt.figure(
        figsize=(8, 5)
    )

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

    print(
        "Emotion distribution graph saved!"
    )


# ======================================
# Plot Feature Importance
# ======================================

def plot_feature_importance(
    model
):

    importance = (
        model.feature_importances_
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        range(len(importance)),
        importance
    )

    plt.title(
        "Feature Importance"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.ylabel(
        "MFCC Features"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png"
    )

    plt.close()

    print(
        "Feature importance graph saved!"
    )


# ======================================
# Main Function
# ======================================

def main():

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    print(
        "\n========== MODEL RESULTS =========="
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    save_results(
        accuracy,
        report
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "Model saved successfully!"
    )

    plot_confusion_matrix(
        y_test,
        predictions
    )

    plot_distribution(
        y
    )

    plot_feature_importance(
        model
    )

    print(
        "\nProject executed successfully!"
    )


if __name__ == "__main__":
    main()
