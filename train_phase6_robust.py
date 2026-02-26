"""
train_phase6_robust.py

This script trains a robust classifier to address class imbalance issues in the wafer
defect dataset. It calculates class weights, freezes the layers of a pre-trained encoder,
builds a robust classification head, trains the model with the class weights, and evaluates
its performance with specific focus on challenging classes.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import os


def train_phase6_robust():
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

    # --- 1. Calculate Class Weights ---
    print("\n--- Calculating Class Weights ---")
    # y_train contains integer labels (0-8)
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    print("Class Weights:")
    for cls, weight in class_weight_dict.items():
        print(f"Class {cls}: {weight:.4f}")

    # --- 2. Model Assembly (Frozen Encoder) ---
    print("\n--- Loading Pre-trained Encoder ---")
    encoder_path = "pretrained_encoder_v2.h5"
    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found.")
        return

    encoder = models.load_model(encoder_path)

    # Freeze Encoder
    encoder.trainable = False
    print("Encoder set to NON-TRAINABLE (Frozen).")

    # Build Classifier Head
    inputs = layers.Input(shape=(60, 60, 1))
    x = encoder(inputs)
    x = layers.Flatten()(x)

    # Robust Head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)  # Additional layer for capacity
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    robust_model = models.Model(inputs, outputs)

    print("\n--- Robust Model Summary ---")
    robust_model.summary()

    # --- 3. Training Setup ---
    print("\n--- Compiling and Training with Class Weights ---")
    # Higher LR allowed since Encoder is frozen
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    robust_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = robust_model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
    )  # APPLY WEIGHTS

    # --- 4. Artifacts & Evaluation ---
    model_path = "robust_classifier.h5"
    robust_model.save(model_path)
    print(f"\nRobust model saved to {os.path.abspath(model_path)}")

    # Evaluate
    test_loss, test_acc = robust_model.evaluate(X_test, y_test, verbose=2)
    print(f"\n--- Final Test Accuracy: {test_acc * 100:.2f}% ---")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Acc")
    plt.plot(history.history["val_accuracy"], label="Validation Acc")
    plt.title("Robust Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Robust Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("robust_training_history.png")
    print("Training history saved to 'robust_training_history.png'")

    # Confusion Matrix & Specific Metrics
    y_pred_probs = robust_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

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

    # Classification Report to get Recall for each class
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Robust Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("robust_confusion_matrix.png")
    print("Confusion matrix saved to 'robust_confusion_matrix.png'")

    # Extract Scratch Recall manually if needed
    # Scratch index is 7 based on label mapping
    scratch_Recall = cm[7, 7] / np.sum(cm[7, :])
    print(f"\nSimple Recall check for 'Scratch': {scratch_Recall * 100:.2f}%")


if __name__ == "__main__":
    train_phase6_robust()
