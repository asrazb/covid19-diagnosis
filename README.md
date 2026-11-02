# Comparing the Performance of Deep Neural Networks as Feature Extractors for COVID-19 Diagnosis

## Overview

This project evaluates pre-trained Convolutional Neural Networks (CNNs) as feature extractors for COVID-19 diagnosis using chest X-ray images.

Deep features extracted from CNN architectures are selected using the Chi-Square (χ²) method and classified with XGBoost.

## Methodology

The proposed approach consists of four main steps:

1. **Image Preprocessing:** Resizing chest X-ray images to 224 × 224 pixels.
2. **Feature Extraction:** Using pre-trained CNN architectures as fixed feature extractors.
3. **Feature Selection:** Selecting the top 1,000 features using Chi-Square.
4. **Classification:** Applying XGBoost to the selected features.

## Classification Tasks

Two classification scenarios are implemented:

- **Binary Classification:** COVID-19 vs. No Findings (5-fold cross-validation).
- **Three-Class Classification:** COVID-19, Pneumonia, and No Findings (80/20 train-test split).

## Results

| Classification Task        | Accuracy |
| -------------------------- | -------: |
| Binary Classification      |    98.4% |
| Three-Class Classification |   87.55% |

DenseNet169 is used as the feature extractor in the documented experiments.

## Technologies

Python · TensorFlow/Keras · Scikit-learn · XGBoost · OpenCV · NumPy · Pandas · Matplotlib · Seaborn

## Project Structure

```text
covid19-diagnosis/
├── README.md
├── binary_classification.ipynb
└── three_class_classification.ipynb
```

## How to Run

1. Open either notebook in Google Colab.
2. Mount Google Drive.
3. Update the dataset paths if necessary.
4. Run the notebook cells sequentially.
