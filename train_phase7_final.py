"""
train_phase7_final.py

This script performs full fine-tuning of the pre-trained encoder along with the
robust classification head. It unfreezes all layers, trains the entire network
with a low learning rate, and evaluates the final phase 7 model's performance
on the wafer defect classification task.
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


def train_phase7_final():
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
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    print("Class Weights:")
    for cls, weight in class_weight_dict.items():
        print(f"Class {cls}: {weight:.4f}")

    # --- 2. Model Assembly (Unfrozen Encoder) ---
    print("\n--- Loading Pre-trained Encoder ---")
    encoder_path = "pretrained_encoder_v2.h5"
    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found.")
        return

    encoder = models.load_model(encoder_path)

    # Unfreeze Encoder
    encoder.trainable = True
    print("Encoder set to TRAINABLE (Unfrozen).")

    # Build Classifier Head
    inputs = layers.Input(shape=(60, 60, 1))
    x = encoder(inputs)
    x = layers.Flatten()(x)

    # Robust Head structure from Phase 6
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    final_model = models.Model(inputs, outputs)

    print("\n--- Phase 7 Model Summary ---")
    final_model.summary()

    # --- 3. Training Setup ---
    print("\n--- Compiling and Training (Full Fine-Tuning) ---")
    # Low LR for fine-tuning
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    final_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = final_model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
    )  # APPLY WEIGHTS

    # --- 4. Artifacts & Evaluation ---
    model_path = "phase7_model.h5"
    final_model.save(model_path)
    print(f"\nFinal model saved to {os.path.abspath(model_path)}")

    # Evaluate
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=2)
    print(f"\n--- Final Test Accuracy: {test_acc * 100:.2f}% ---")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Acc")
    plt.plot(history.history["val_accuracy"], label="Validation Acc")
    plt.title("Phase 7 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Phase 7 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("phase7_history.png")
    print("Training history saved to 'phase7_history.png'")

    # Confusion Matrix
    y_pred_probs = final_model.predict(X_test)
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

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Phase 7 Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("phase7_cm.png")
    print("Confusion matrix saved to 'phase7_cm.png'")

    # Scratch Recall calc
    # Index 7 = Scratch
    scratch_recall = cm[7, 7] / np.sum(cm[7, :])
    print(f"\nPhase 7 Scratch Recall: {scratch_recall * 100:.2f}%")


if __name__ == "__main__":
    train_phase7_final()
