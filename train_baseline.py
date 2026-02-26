"""
train_baseline.py

This script defines, trains, and evaluates a baseline Convolutional Neural Network (CNN)
model for classifying wafer defect patterns. It loads the preprocessed data, builds the CNN
architecture, trains it, saves the trained model, and generates evaluation plots such as
training history and a confusion matrix.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def train_baseline():
    print("--- Loading Data ---")
    try:
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")
        print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 1. Data Preparation ---
    # Reshape X to (N, 60, 60, 1) to add channel dimension
    X_train = X_train.reshape((-1, 60, 60, 1))
    X_test = X_test.reshape((-1, 60, 60, 1))

    print(f"Reshaped X_train: {X_train.shape}")
    print(f"Reshaped X_test: {X_test.shape}")

    num_classes = 9

    # --- 2. Model Architecture ---
    print("\n--- Building Model ---")
    model = models.Sequential(
        [
            # Layer 1
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(60, 60, 1)),
            layers.MaxPooling2D((2, 2)),
            # Layer 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Layer 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Convert to 1D
            layers.Flatten(),
            # Dense Layers
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),  # Dropout to prevent overfitting
            # Output Layer
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    # --- 3. Training Setup ---
    print("\n--- Compiling and Training ---")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test)
    )

    # --- 4. Artifacts & Outputs ---
    # Save Model
    str_model_name = "baseline_cnn.h5"
    model.save(str_model_name)
    print(f"\nModel saved to {os.path.abspath(str_model_name)}")

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n--- Final Test Accuracy: {test_acc * 100:.2f}% ---")

    # Plot 1: Training Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Acc")
    plt.plot(history.history["val_accuracy"], label="Validation Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("training_history.png")
    print("Training history plot saved to 'training_history.png'")

    # Plot 2: Confusion Matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    # We need to manually reconstruct the label mapping or just use numbers if passing isn't easy
    # Or strict mapping from preprocess:
    # {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Near-full': 5, 'Random': 6, 'Scratch': 7, 'none': 8}
    labels = [
        "Center",
        "Donut",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "Near-full",
        "Random",
        "Scratch",
        "none",
    ]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix plot saved to 'confusion_matrix.png'")


if __name__ == "__main__":
    train_baseline()
