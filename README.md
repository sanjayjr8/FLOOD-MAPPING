# Bayesian U-Net for Uncertainty-Aware Flood Mapping from SAR Imagery

<div align="center">
  <img src="https://storage.googleapis.com/maker-media-experiment/20240509_110534_901300_sanjay_j_0.gif" alt="Project Animation" width="600"/>
</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

This repository contains the official implementation for the paper: **"Bayesian U-Net–Enabled Pixel-Level Flood Mapping from Synthetic Aperture Radar Images."** We introduce a deep learning framework that not only achieves high-accuracy flood segmentation but also quantifies its own predictive uncertainty, addressing a critical gap in current disaster management tools.

---

## 🎯 The Problem: The Illusion of Certainty

State-of-the-art deep learning models for flood mapping are powerful but flawed. They are **deterministic**, meaning they produce a single, overconfident flood map. They are forced to make a "yes" or "no" decision for every pixel, even when the data is ambiguous.

In the real world, this is a major risk. A model can be **confidently wrong** about a wet road or a building's shadow, leading to flawed decision-making during a crisis.

<div align="center">
  <img src="https://storage.googleapis.com/maker-media-experiment/20240509_105436_531604_sanjay_j_0.jpg" alt="Problem Statement" width="700"/>
</div>

---

## ✨ Our Solution: From Prediction to Insight

Our work introduces a **Bayesian U-Net** that moves beyond simple prediction to provide **actionable insight**. By leveraging Monte Carlo Dropout, our model performs multiple stochastic forward passes to approximate its own uncertainty.

The result is a dual-output system:
1.  **A high-accuracy Flood Map** (the mean of all predictions).
2.  An **Uncertainty Map** (the standard deviation), which highlights areas where the model is "not sure."

This transforms the model from a black box into a transparent, trustworthy decision-support tool.

<div align="center">
  <img src="https://storage.googleapis.com/maker-media-experiment/20240509_110129_023476_sanjay_j_0.jpg" alt="Methodology" width="800"/>
</div>

---

## 🛰️ The Dataset: HISEA Flooding Dataset

This project utilizes the **HISEA flooding dataset**, a benchmark collection specifically curated for developing and testing flood segmentation models on SAR imagery.

* **Specialty:** The dataset's primary advantage is its focus on **Synthetic Aperture Radar (SAR)** imagery. Unlike optical images, SAR can penetrate clouds and operate day or night, making it the only reliable data source during active storm and flood events. It presents unique challenges like **speckle noise** and **ambiguous backscatter**, which our model learns to navigate.
* **Preprocessing:** All images undergo standardization (mean/std normalization) to ensure stable and efficient model training.

---

## 🚀 Training & Performance

The model was trained for **10 epochs** using a combined **BCE and Dice Loss** function with the Adam optimizer. The best model checkpoint was saved based on the peak validation IoU score.

### Quantitative Results

The model achieved a strong proof-of-concept performance, demonstrating its capability to accurately segment floodwater while learning to quantify uncertainty.

| Metric | Best Value | Epoch Achieved |
| :--- | :---: | :---: |
| **Validation IoU** | **0.6185** | **10** |
| Validation Loss | 0.3480 | 10 |

The training history below shows stable learning and indicates that performance can be further improved with extended training.

<p align="center">
  <img src="https://i.imgur.com/your_loss_plot_url.png" alt="Loss vs Epochs" width="48%">
  <img src="https://i.imgur.com/your_iou_plot_url.png" alt="IoU vs Epochs" width="48%">
</p>
*(Note: Replace the placeholder URLs with actual images of your plots.)*

### Qualitative Results

Visual analysis confirms the model's dual capabilities. The uncertainty maps consistently highlight the most challenging regions, such as flood boundaries, noisy areas, and complex terrain.

<div align="center">
  <img src="https://i.imgur.com/your_results_grid_url.png" alt="Qualitative Results" width="800"/>
</div>
*(Note: Replace the placeholder URL with the 4-panel image from your paper.)*

---

## 💡 Novelty & Contribution

While many models chase higher accuracy scores, our work pioneers the integration of **trustworthiness** into the flood mapping process.

1.  **Bridging the Application Gap:** We are the first to rigorously apply and validate a Bayesian U-Net specifically for the **noisy and ambiguous domain of SAR flood imagery**, a field where it has been critically underutilized.
2.  **Shifting the Goal:** We shift the paradigm from "accuracy-only" to **"accuracy + confidence."** A model that knows when it's unsure is fundamentally more useful in a crisis than a slightly more accurate one that operates as a black box.
3.  **Operational Value:** Our dual-output system provides an end-to-end framework for **actionable intelligence**, allowing emergency responders to trust the predictions and prioritize verification efforts where they are needed most.

---

## 🛠️ Getting Started

### Libraries Used

To run this project, you will need Python 3.9+ and the following core libraries:

* **PyTorch** (`torch`, `torchvision`)
* **NumPy**
* **Rasterio** (for handling `.tif` satellite images)
* **Albumentations** (for image augmentation)
* **Matplotlib** (for plotting)
* **OpenCV** (`opencv-python`)
* **Scikit-learn** (`sklearn`)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the HISEA dataset and place it in the appropriate directory.

4.  Run the Jupyter Notebook `capstone1.ipynb` to train the model and generate results.

---

## 📄 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{your_citation_key,
  author    = {Sanjay, J. and Miruthula, SK and Vijay Krishna Ji, V},
  title     = {Bayesian U-Net–Enabled Pixel-Level Flood Mapping from Synthetic Aperture Radar Images},
  booktitle = {Conference Name},
  year      = {2025},
}
