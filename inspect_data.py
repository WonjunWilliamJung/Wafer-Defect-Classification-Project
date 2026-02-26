"""
inspect_data.py

This script is responsible for loading the Wafer Defect dataset (LSWMD.pkl) and
performing exploratory data analysis. It prints dataframe information, checks for
missing values, analyzes the distribution of failure types, and generates
visualizations of randomly selected wafer maps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def inspect_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_pickle(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Print dataframe info, columns, and shape
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Columns ---")
    print(df.columns.tolist())
    print("\n--- Shape ---")
    print(df.shape)

    # 3. Check for missing values and distribution of 'failureType'
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    if "failureType" in df.columns:
        print("\n--- Failure Type Distribution ---")
        # Ensure failureType is readable (it might be encoded or arrays)
        # Check one value to see format
        sample_ft = df["failureType"].iloc[0]
        print(f"Sample failureType format: {sample_ft} (Type: {type(sample_ft)})")

        # Optimize: Convert to string label directly to avoid value_counts issues with arrays
        try:
            # Assumes format is likely [['label']] or similar based on sample
            df["failureType_clean"] = df["failureType"].apply(
                lambda x: (
                    x[0][0]
                    if (
                        isinstance(x, (np.ndarray, list))
                        and len(x) > 0
                        and len(x[0]) > 0
                    )
                    else str(x)
                )
            )
            print(df["failureType_clean"].value_counts())
        except Exception as e:
            print(f"Error processing failureType: {e}")
    else:
        print("\n'failureType' column not found.")

    # 4. Visualize 5 random wafer maps
    if "waferMap" in df.columns:
        print("\n--- Generating Visualization ---")
        num_samples = 5
        indices = random.sample(range(len(df)), num_samples)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for i, idx in enumerate(indices):
            row = df.iloc[idx]
            wafer_map = row["waferMap"]

            # Handling failure label for title
            if "failureType" in df.columns:
                failure_label = row["failureType"]
                # Clean up label if it's nested structure common in this dataset
                if (
                    isinstance(failure_label, (np.ndarray, list))
                    and len(failure_label) > 0
                ):
                    if hasattr(failure_label[0], "__getitem__"):  # Nested
                        failure_label = failure_label[0][0]
                    else:
                        failure_label = failure_label[0]
            else:
                failure_label = "Unknown"

            axes[i].imshow(wafer_map)
            axes[i].set_title(f"Idx: {idx}\n{failure_label}")
            axes[i].axis("off")

        output_file = "wafer_samples.png"
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Visualization saved to {os.path.abspath(output_file)}")
    else:
        print("\n'waferMap' column not found, skipping visualization.")


if __name__ == "__main__":
    file_path = "LSWMD.pkl"
    if os.path.exists(file_path):
        inspect_data(file_path)
    else:
        print(f"File not found: {file_path}")
