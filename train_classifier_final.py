"""
train_classifier_final.py

This script trains a final hybrid classifier by loading a pre-trained encoder and attaching
a new classification head. The entire model (encoder + classifier) is fine-tuned on the
labeled wafer defect data, saved, and evaluated by plotting training history and a confusion matrix.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def train_classifier_final():
    print("--- Loading Labeled Data ---")
    try:
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")
        print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Reshape
    X_train = X_train.reshape((-1, 60, 60, 1))
    X_test = X_test.reshape((-1, 60, 60, 1))

    num_classes = 9

    # --- Model Assembly ---
    print("\n--- Loading Pre-trained Encoder ---")
    encoder_path = "pretrained_encoder_v2.h5"
    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found.")
        return

    # Load encoder
    # Note: If saved as a Model object (inputs, outputs), we can just load it.
    encoder = models.load_model(encoder_path)
    # Encoder output is the 'latent_output' layer which is (7, 7, 128) before flatten?
    # Or whatever the pooling gave. 60->30->15->8 (padding same 60/2=30, 30/2=15, 15/2=7.5->8)
    # Let's check summary when building.

    print("Encoder Summary:")
    encoder.summary()

    # Build Classifier on top
    # We want to treat 'encoder' as a layer/block.
    # Input
    inputs = layers.Input(shape=(60, 60, 1))
    x = encoder(inputs)  # Apply encoder
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    final_model = models.Model(inputs, outputs)

    print("\n--- Final Hybrid Model Summary ---")
    final_model.summary()

    # --- Training Strategy ---
    # Fine-tuning: Updates all weights (Encoder + Classifier)
    print("\n--- Compiling and Fine-tuning ---")
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # Low LR
    final_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = final_model.fit(
        X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test)
    )

    # --- Artifacts & Evaluation ---
    # Save Model
    model_path = "final_model.h5"
    final_model.save(model_path)
    print(f"\nFinal model saved to {os.path.abspath(model_path)}")

    # Evaluate
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=2)
    print(f"\n--- Final Test Accuracy: {test_acc * 100:.2f}% ---")

    # Compare with Baseline (hardcoded baseline value from previous step usually, or just print current)
    print(f"Baseline Accuracy (Reference): ~82.87%")

    # Plot Training History
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Acc")
    plt.plot(history.history["val_accuracy"], label="Validation Acc")
    plt.title("Final Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Final Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("final_training_history.png")
    print("Training history saved to 'final_training_history.png'")

    # Confusion Matrix
    y_pred_probs = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
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
    plt.title("Final Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png")
    print("Confusion matrix saved to 'final_confusion_matrix.png'")


if __name__ == "__main__":
    train_classifier_final()
