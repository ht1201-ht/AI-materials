
# Transformer-Based Regression for Machine-Learning-Guided Phosphor Optimization

This repository provides the code for the regression modeling of compositional–property relationships in Mo(IV)-activated halide phosphors using a Transformer neural network, as described in our work:

**Machine Learning-Guided Design and Bandgap Engineering of Mo(Ⅳ)-Activated Halide Phosphors for Efficient NIR Emission and AI-Augmented Imaging**

## 1. Overview

This project implements a PyTorch-based Transformer regression model to quantitatively predict photoluminescence (PL) intensity from experimental compositional variables (B, C co-doping ratios). The model enables data-driven optimization and rational design of NIR-emissive materials by automatically extracting non-linear and cross-term dependencies within the compositional space.

## 2. Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.11
* pandas, numpy, matplotlib, scikit-learn, openpyxl

Install dependencies via:

```bash
pip install torch pandas numpy matplotlib scikit-learn openpyxl
```

## 3. Data Preparation

Prepare an Excel file named `.xlsx` in the project root directory, with the following columns:

| A     | B    |  C   |
| ----- | ---- | ---- |
| 0.015 | 0.41 | 5283 |
| ...   | ...  | ...  |

* Ensure the file contains valid numeric data, and no missing values.

## 4. Running the Code

Execute the main script to perform model training, evaluation, and compositional optimization:

```bash
python transformer_regression_gpu.py
```

The workflow includes:

* Automated data preprocessing (including feature engineering and normalization)
* Model training and test-set evaluation
* Global prediction over the B/C co-doping space to identify the optimal composition for maximal PL intensity

## 5. Outputs

* Training loss is reported every 20 epochs.
* Final model performance (RMSE and R² on the test set) is printed in the terminal.
* The optimal B and C doping ratios, as predicted by the trained model, are reported.
* A heatmap (`transformer_prediction_heatmap.png`) is generated, visualizing predicted PL intensity across the full composition space, with the optimal point highlighted.

## 6. Notes

* GPU acceleration is supported and automatically enabled if available.
* All model parameters (layer dimensions, epochs, batch size, learning rate) can be modified in the script.
* If the program reports data errors, please check the format and content of the Excel file.

## 7. Citation

If you use this code or data in your research, please cite our paper (bibtex in main README).

---

**For any questions or feedback, please open an issue or contact the corresponding author.**


