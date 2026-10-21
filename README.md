# AI Medical Image Diagnosis & Explainability
## Chest X-ray Pneumonia Detection Using Deep Learning

An educational deep-learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a custom PyTorch CNN. The project also explores Grad-CAM visual explanations, preprocessing choices, classification thresholds, and potential sources of model error.

> **Disclaimer:** This is an educational research prototype, not a clinically validated diagnostic system. It must not be used to make medical decisions.

## Project Overview

The goal is to build and evaluate a chest X-ray classification pipeline while investigating whether the model's predictions are reliable beyond the validation dataset.

### Workflow

1. Load chest X-ray images.
2. Preprocess images using grayscale conversion, resizing, and padding.
3. Train a custom convolutional neural network (CNN).
4. Evaluate predictions using classification metrics and ROC-AUC.
5. Investigate classification thresholds using validation data.
6. Explore Grad-CAM visualizations and padding sensitivity.

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Image Processing:** Pillow, torchvision
- **Data Analysis:** NumPy, Pandas
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib
- **Explainability:** Grad-CAM

## Dataset

The project uses a chest X-ray dataset with two classes:

- `NORMAL` — images labeled as normal
- `PNEUMONIA` — images labeled as pneumonia

The training directory contains 5,216 images. A stratified split was used to create training and validation subsets. The official test directory contains 624 images: 234 NORMAL and 390 PNEUMONIA.

## Model

A custom CNN built with PyTorch uses convolutional layers, ReLU activations, max-pooling, and a final linear classifier. The model accepts single-channel, 224 × 224 images and outputs a pneumonia logit.

## Experiments

The project compares three preprocessing strategies:

- Direct resizing to 224 × 224
- Aspect-ratio-preserving resizing with padding
- Randomized padding during training

It also explores validation-based threshold selection and Grad-CAM visualizations of selected predictions.

## Evaluation

The model achieved approximately 97% validation accuracy, but performance on the official test set was substantially lower. For the randomized-padding model at a 0.50 threshold, test accuracy was approximately 76.9%, with a ROC-AUC of 0.943.

This gap indicates that validation accuracy alone is not sufficient to establish generalization. The official test set was also used during exploratory comparisons, so it should not be considered an untouched, independent final evaluation.

Threshold selection should be performed using validation data, not by optimizing results on the test set.

## Explainability and Error Analysis

The project includes exploratory analyses of:

- Grad-CAM activation maps for selected images
- False-positive NORMAL predictions
- Sensitivity to modifications of image padding
- Validation classification metrics across probability thresholds

These analyses are preliminary. Grad-CAM highlights model activations but does not prove which features caused a prediction. Padding interventions create artificial inputs and do not establish the cause of classification errors.

### Generated Reports

- `reports/figures/false_positive_gradcam.png`
- `reports/figures/padding_sensitivity_changes.png`
- `reports/figures/validation_threshold_curve.png`
- `reports/marker_sensitivity_top5.csv`
- `reports/validation_threshold_results.csv`

## Limitations and Future Work

- Investigate the validation-to-test performance gap.
- Check for dataset and preprocessing distribution differences.
- Evaluate performance across different random seeds and splits.
- Improve class-wise evaluation and error analysis.
- Validate explainability findings across a larger, representative sample.
- Evaluate on a genuinely independent dataset before making stronger generalization claims.

## Author

**Anubhav Singh**  
B.Tech, Computer Science and Engineering

GitHub: [anubhavsingh23906-code](https://github.com/anubhavsingh23906-code)

---

**For research and learning purposes only. Not intended for clinical use.**