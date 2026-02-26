# Wafer Defect Classification Project

This project focuses on identifying and classifying defect patterns in semiconductor wafer maps using deep learning. It uses the `LSWMD.pkl` dataset and implements a multi-stage approach, evolving from a simple baseline Convolutional Neural Network (CNN) to an advanced self-training model with attention mechanisms.

## Project Structure

1. **Data Inspection & Preprocessing**:
   - `inspect_data.py`: Loads the dataset, checks for missing values, analyzes class distributions, and creates visualizations.
   - `preprocess_data.py`: Cleans labels, balances classes, resizes images, and prepares normalized `.npy` array files for training.

2. **Model Training Phases**:
   - `train_baseline.py`: A basic CNN model for initial defect classification.
   - `train_autoencoder_final.py`: A convolutional autoencoder trained on both labeled and unlabeled data for feature extraction.
   - `train_classifier_final.py`: A hybrid classifier training a classification head on top of the pre-trained encoder.
   - `train_phase6_robust.py`: Addresses class imbalance by freezing the encoder, generating class weights, and training a robust head.
   - `train_phase7_final.py`: Full fine-tuning of the entire robust model architecture.
   - `train_phase8_uncertainty.py`: Implements self-training with pseudo-labels and trains a CBAM-CNN student model.

## Usage

1. **Pre-requisites**: Ensure you have Python with `pandas`, `numpy`, `tensorflow`/`keras`, `opencv-python`, `scikit-learn`, `matplotlib`, and `seaborn` installed.
2. **Data**: Place the `LSWMD.pkl` dataset file in the root directory.
3. **Execution**: Run the scripts in order, starting from data inspection, preprocessing, baseline training, and then moving towards advanced models (phases 6-8).

## Outputs

Each training script outputs its model weights as `.h5` files, alongside visual metrics such as training history plots and confusion matrices saved as `.png` files to track the improvements iteration by iteration.
