# Chest X-ray Pneumonia Detection & Explainability

A deep-learning research project that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** categories using a custom Convolutional Neural Network (CNN) built with PyTorch.

Beyond classification, this project investigates model behavior through **Grad-CAM explainability, preprocessing sensitivity, marker-region interventions, and corner suppression experiments**.

> **Disclaimer:** This is an educational research prototype, not a clinically validated diagnostic system. It must not be used for medical diagnosis or treatment decisions.

## Project Overview

The objective is not only to train a CNN, but also to investigate how its predictions behave under different evaluation conditions and controlled image modifications.

### Key Highlights

- Custom PyTorch CNN for binary chest X-ray classification.
- Grayscale image preprocessing with aspect-ratio-preserving resizing and padding.
- Validation-based classification threshold analysis.
- Grad-CAM visualizations for model explainability.
- Marker-region sensitivity experiments using controlled image modifications.
- Corner suppression fine-tuning to investigate potential border-related sensitivity.
- Evaluation of the validation-to-test performance gap.

## Tech Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Image Processing | Pillow, torchvision |
| Data Analysis | NumPy, Pandas |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib |
| Explainability | Grad-CAM |
| Development | Jupyter Notebook |

## Dataset

The project uses a chest X-ray dataset with two classes:

- `NORMAL` â€” images labeled as normal.
- `PNEUMONIA` â€” images labeled as pneumonia.

| Dataset portion | Images |
|---|---:|
| Training source | 5,216 |
| Training split | 4,172 |
| Validation split | 1,044 |
| Official test set | 624 |
| Total | 5,840 |

The training source was divided using a stratified 80:20 split.

The official test set contains 234 NORMAL and 390 PNEUMONIA images.

Dataset files are not included in this repository.

## Model Architecture

A custom CNN implemented in PyTorch.

- Input: single-channel grayscale image.
- Image dimensions: 224 Ã— 224.
- Convolutional layers with 16 and 32 channels.
- ReLU activation functions.
- Max-pooling layers.
- Fully connected binary classification layer.
- Output: pneumonia logit, converted to a probability for prediction.

The model uses `BCEWithLogitsLoss` for binary classification.

## Evaluation Results

The randomized-padding model was evaluated at different classification thresholds.

| Metric | Validation (0.30) | Official test (0.30) |
|---|---:|---:|
| Accuracy | 97.13% | 74.04% |
| ROC-AUC | 0.9948 | 0.9429 |

At a 0.50 threshold, official test accuracy was approximately **76.92%**, with a ROC-AUC of approximately **0.943**.

The substantial validation-to-test accuracy gap highlights the importance of investigating generalization rather than relying on validation accuracy alone.

**Evaluation caveat:** The official test set was used repeatedly during exploratory analysis. Therefore, these results should not be interpreted as an untouched, independent final evaluation. Threshold selection should be based on validation data, not optimized against the test set.

## Explainability and Experiments

### 1. Grad-CAM Explainability

Grad-CAM was used to visualize regions associated with model activations for selected predictions.

The analysis includes:
- Correct NORMAL predictions.
- Correct PNEUMONIA predictions.
- NORMAL false-positive predictions.
- Region-level activation comparisons.

Visual reports:
- [Correct NORMAL Grad-CAM](reports/gradcam_visualizations/correct_normal.png)
- [Correct PNEUMONIA Grad-CAM](reports/gradcam_visualizations/correct_pneumonia.png)
- [NORMAL false-positive Grad-CAM](reports/gradcam_visualizations/normal_false_positives.png)
- [Complete Grad-CAM report (PDF)](reports/gradcam_visualizations/gradcam_complete_visual_report.pdf)

Grad-CAM provides an exploratory view of model activations. It does not establish causal explanations for predictions.

### 2. Marker Sensitivity Experiments

Controlled image modifications were used to investigate model sensitivity around manually identified marker regions.

The experiments compared marker-region logit changes against control regions, including:
- Local-mean replacement.
- Blur-based modifications.
- Random control regions.
- Additional robustness comparisons.

In the selected 12-image analysis, marker-region changes exceeded the control 95th percentile in 12/12 images for local-mean replacement and 11/12 images for blur.

These findings suggest elevated sensitivity to the selected regions in this sample. They do not prove that the model causally relies on medical image markers.

### 3. Corner Suppression Fine-Tuning

A controlled experiment compared a baseline fine-tuned model with a model trained using random border-patch replacement on a proportion of training images.

The experiment used:
- Five fine-tuning epochs.
- The same initial checkpoint.
- Matching training settings.
- Validation performance comparisons.
- Marker sensitivity comparisons.

The two models achieved nearly identical validation performance, and border suppression did not meaningfully reduce marker sensitivity in the selected analysis.

This experiment does not establish that all corner or border-related shortcuts have been eliminated.

Visual results:
- [Overall marker sensitivity](models/corner_suppression_experiment/final_plots/overall_marker_sensitivity.png)
- [Marker control exceedance](models/corner_suppression_experiment/final_plots/marker_control_exceedance.png)
- [Local-mean group sensitivity](models/corner_suppression_experiment/final_plots/group_sensitivity_local_mean.png)
- [Blur group sensitivity](models/corner_suppression_experiment/final_plots/group_sensitivity_blur.png)

## Repository Structure

```text
chest-xray-ai/
â”‚
â”œâ”€â”€ app/
â”œâ”€â”€ data/
â”œâ”€â”€ models/
â”‚   â””â”€â”€ corner_suppression_experiment/
â”‚       â””â”€â”€ final_plots/
â”‚
â”œâ”€â”€ notebooks/
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ figures/
â”‚   â””â”€â”€ gradcam_visualizations/
â”‚       â””â”€â”€ marker_inspection/
â”‚
â”œâ”€â”€ results/
â”œâ”€â”€ src/
â”‚
â”œâ”€â”€ chest_xray_pneumonia_analysis.ipynb
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â””â”€â”€ .gitignore
```

Dataset files, virtual environments, and model checkpoint binaries are excluded through `.gitignore`.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anubhavsingh23906-code/chest-xray-ai.git
cd chest-xray-ai
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the dataset in the expected directory structure used by the notebook and project pipeline. Dataset files are not distributed with this repository.

### 5. Explore the analysis

Open:

```bash
jupyter notebook chest_xray_pneumonia_analysis.ipynb
```

Run the notebook cells in order after preparing the dataset and environment.

## Limitations and Future Work

- Investigate the validation-to-test performance gap.
- Evaluate on a genuinely independent external dataset.
- Repeat experiments across multiple random seeds and data splits.
- Expand marker-region analysis to a larger, independently selected sample.
- Investigate possible dataset and preprocessing biases.
- Improve class-wise evaluation and error analysis.
- Validate explainability observations with more rigorous methods.

## Author

**Anubhav Singh**

B.Tech â€” Computer Science and Engineering

- GitHub: [anubhavsingh23906-code](https://github.com/anubhavsingh23906-code)
- Project Repository: [chest-xray-ai](https://github.com/anubhavsingh23906-code/chest-xray-ai)

---

*For educational and research purposes only. Not intended for clinical use.*
