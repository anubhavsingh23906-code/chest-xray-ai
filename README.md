\# AI Medical Image Diagnosis \& Explainability



\## Chest X-ray Disease Detection



An educational AI project for detecting diseases from chest X-ray images

using deep learning and explaining predictions using Grad-CAM.



\## Pipeline



Chest X-ray

→ Preprocessing

→ Deep Learning Model

→ Disease Prediction

→ Confidence Score

→ Grad-CAM

→ Visual Explanation



\## Tech Stack



\- Python

\- PyTorch

\- NumPy

\- Pandas

\- Scikit-learn

\- Pillow

\- Matplotlib

\- FastAPI

\- Grad-CAM



\## Important



This project is an educational/research prototype.

It is NOT a medical diagnostic tool.

## Padding Sensitivity Analysis

### Objective
Investigate whether the CNN's pneumonia predictions are sensitive to artificial padding introduced during aspect-ratio-preserving image resizing.

### Method
Selected 20 false-positive NORMAL images from the official test set. For each image, the model's original pneumonia probability was compared with its probability after replacing only the geometrically identified padding pixels with the mean intensity of the X-ray image content.

### Results
- Images analyzed: 20
- Images with lower pneumonia probability after modification: 15/20
- Mean original pneumonia probability: 87.95%
- Mean modified pneumonia probability: 81.07%
- Mean probability change: -6.87 percentage points

### Interpretation and Limitations
The results suggest that the model is sensitive to padding modifications on this selected subset. This intervention does not establish that padding causes the original classification errors because replacing padding with gray pixels creates an artificial input. The sample contains only selected false positives and is not representative of the entire test set. Further experiments using controlled padding values, diverse test images, and independent evaluation are needed.

### Visualizations
- `reports/figures/false_positive_gradcam.png`
- `reports/figures/padding_sensitivity_changes.png`

**Disclaimer:** This is an educational research prototype, not a clinically validated diagnostic system.