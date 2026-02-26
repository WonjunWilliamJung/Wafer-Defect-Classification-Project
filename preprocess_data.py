"""
preprocess_data.py

This script handles the preprocessing of the Wafer Defect dataset. It cleans the labels,
balances the classes by sampling, resizes the wafer map images, normalizes the pixel values,
and splits the data into training and testing sets. The preprocessed data is then saved
as NumPy arrays (.npy files) for model training.
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import os


def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_pickle(file_path)
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 1. Load & Clean Labels ---
    print("\n--- Cleaning Labels ---")
    # Step A: Drop empty labels
    # Assumes failureType is a list/array
    # We want to keep rows where failureType is NOT empty list and NOT empty array
    # A safe way is to check length > 0 if iterable

    # Helper to check if empty
    def is_empty_label(x):
        if isinstance(x, (np.ndarray, list)):
            return len(x) == 0
        return False

    initial_count = len(df)
    df = df[~df["failureType"].apply(is_empty_label)]
    print(f"Dropped rows with empty labels. Count: {initial_count} -> {len(df)}")

    # Step B: Convert list to string
    # Assumes format [['Loc']] -> 'Loc'
    def clean_label(x):
        if isinstance(x, (np.ndarray, list)) and len(x) > 0:
            if isinstance(x[0], (np.ndarray, list)) and len(x[0]) > 0:
                return str(x[0][0])
            return str(x[0])
        return str(x)

    df["failureType"] = df["failureType"].apply(clean_label)

    # Filter out 'none' if it's not the string 'none' (just in case of weird labeling)
    # The requirement says: Separate 'none' vs 'Defect'
    print(f"Unique classes found: {df['failureType'].unique()}")

    # --- Step C (Class Balancing) ---
    print("\n--- Balancing Classes ---")
    df_none = df[df["failureType"] == "none"]
    df_defect = df[df["failureType"] != "none"]

    print(f"Original 'none' count: {len(df_none)}")
    print(f"Defect count: {len(df_defect)}")

    # Randomly sample exactly 2,000 instances from 'none'
    if len(df_none) >= 2000:
        df_none_sampled = df_none.sample(n=2000, random_state=42)
    else:
        print("Warning: 'none' class has fewer than 2000 samples. Using all available.")
        df_none_sampled = df_none

    df_balanced = pd.concat([df_none_sampled, df_defect])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced Dataset Shape: {df_balanced.shape}")
    print(df_balanced["failureType"].value_counts())

    # --- 2. Image Preprocessing (Resizing) ---
    print("\n--- Resizing Images ---")
    target_size = (60, 60)

    def resize_wafer(wafer_map):
        return cv2.resize(wafer_map, target_size, interpolation=cv2.INTER_NEAREST)

    # Process in a loop or apply (loop might be better for progress or debugging if it crashes)
    # Converting column to a list of arrays then resizing is usually fast enough
    resized_maps = []

    # For visualization later
    sample_indices = [0, 100]  # Just random indices
    original_samples = []
    resized_samples = []

    # Get sample original maps before processing all
    if len(df_balanced) > 0:
        original_samples.append(df_balanced.iloc[0]["waferMap"])
        if len(df_balanced) > 100:
            original_samples.append(df_balanced.iloc[100]["waferMap"])
        else:
            original_samples.append(df_balanced.iloc[-1]["waferMap"])

    # Batch resize
    # Convert dataframe column to numpy array of objects might speed up iteration?
    # Just simple apply is fine for ~25k-30k rows (defect is ~25k + 2k none)
    # actually defects are total ~25k + 147k none.
    # Total balanced size will be ~27k. Fast enough.

    # df_balanced['waferMap'] contains numpy arrays
    resized_maps = df_balanced["waferMap"].apply(resize_wafer).tolist()

    if len(resized_maps) > 0:
        resized_samples.append(resized_maps[0])
        if len(resized_maps) > 100:
            resized_samples.append(resized_maps[100])
        else:
            resized_samples.append(resized_maps[-1])

    X = np.array(resized_maps)

    # --- 3. Prepare Training Data ---
    print("\n--- Preparing Training Data ---")
    # Normalize X (but keep integer 0, 1, 2 if user meant "integer values"?)
    # User said: "Normalize the resized wafer maps." but also "keep values as integers (0, 1, 2)".
    # Usually for CNN we normalize to [0,1] or similar.
    # User instruction: "Normalize the resized wafer maps. (Optionally, one-hot encode... but for now, keep them as (N, 60, 60))"
    # Actually, if we normalize, they become floats.
    # But later it says "keep values as integers (0, 1, 2)" in the context of RESIZING logic (to avoid interpolation artifacts).
    # Step 3 says "Normalize...".
    # I will cast to float32 and divide by max value (which is likely 2) to get 0-1 range?
    # Or maybe just keep them as is if we want to use Embedding layer?
    # "CNN training" usually implies float inputs.
    # I'll standard max-scaling: X = X / 2.0 (since max is 2 for defect area usually).
    # Let's check max value first.

    # But wait, user said "keep values as integers (0, 1, 2)" in specific context of RESIZING interpolation.
    # So the resizing output is int. The normalization step comes AFTER.
    # Let's normalize to [0, 1] range: 0->0, 1->0.5, 2->1.0 ? Or just keep as float.
    # I will normalize by just casting to float for now, maybe divide by 2?
    # Standard practice for categorical pixels in CNN is one-hot, but user said "keep them as (N,60,60)".
    # So I will just leave them as integers? Or Normalize to 0-1?
    # "Normalize the resized wafer maps" -> usually implies float.
    # I will do X = X / 2.0 to put in [0, 1].

    # Just to be safe and standard:
    X = X.astype("float32") / 2.0

    # Encode y
    y = df_balanced["failureType"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # --- 4. Verify & Output ---
    print("\n--- Verification ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Label Mapping
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print("\nLabel Mapping:")
    print(label_mapping)

    # Counts in final set
    print("\nClass Counts in Balanced Dataset:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    count_dict = dict(zip(le.inverse_transform(unique), counts))
    print(count_dict)

    # Visual Check
    print("\n--- Generating Visual Check ---")
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Sample 1
    axes[0, 0].imshow(original_samples[0])
    axes[0, 0].set_title(f"Original 1 (Shape: {original_samples[0].shape})")
    axes[0, 1].imshow(resized_samples[0])
    axes[0, 1].set_title(f"Resized 1 (60, 60)")

    # Sample 2
    axes[1, 0].imshow(original_samples[1])
    axes[1, 0].set_title(f"Original 2 (Shape: {original_samples[1].shape})")
    axes[1, 1].imshow(resized_samples[1])
    axes[1, 1].set_title(f"Resized 2 (60, 60)")

    plt.tight_layout()
    plt.savefig("preprocessing_check.png")
    print("Verification image saved to 'preprocessing_check.png'")

    # --- Save Data ---
    print("\n--- Saving Data ---")
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    print("Data saved to .npy files.")


if __name__ == "__main__":
    file_path = "LSWMD.pkl"
    if os.path.exists(file_path):
        preprocess_data(file_path)
    else:
        print(f"File not found: {file_path}")
