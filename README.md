
# 🏗️ Concrete crack detection using a convolutional variational autoencoder (ConvVAE) for anomaly detection 🔍.
## 🎯 Project Focus & Impact

This project addresses a critical challenge in infrastructure maintenance: **automated and objective concrete crack detection**. By leveraging advanced deep learning techniques, this system provides a robust solution to identify structural anomalies, offering a more efficient and reliable alternative to traditional manual inspections.

The **main idea** behind this project is to train a model exclusively on **"normal" (non-cracked) concrete images** and then enable it to **identify anomalies, which in this context are cracks**. This **anomaly detection paradigm** is implemented using a **Convolutional Variational Autoencoder (ConvVAE)**. The VAE learns the intricate patterns of healthy surfaces, and any new image that deviates significantly (indicated by high reconstruction error) is flagged as an anomaly, signifying the presence of a crack. This approach is particularly valuable for real-world applications where obtaining extensive labeled data for rare events or diverse crack types is challenging.

## ✨ Key Skills & Features Demonstrated

This project showcases expertise in:

  * **Deep Learning & Computer Vision:** Designing, training, and evaluating custom neural network architectures (ConvVAE).
  * **Anomaly Detection:** Implementing a robust system capable of identifying critical defects without relying on extensive anomalous training data.
  * **Data Engineering & MLOps Practices:** Rigorous data handling, partitioning (including creating truly unseen external test sets), and data leakage prevention.
  * **Model Evaluation & Interpretability:** Comprehensive performance assessment using industry-standard metrics and advanced visualization for model understanding.
  * **PyTorch Proficiency:** Building and deploying custom `Dataset` and `DataLoader` pipelines, defining custom model architectures, and managing the training lifecycle.

### Core Features:

  * **Automated Data Handling:** Programmatic dataset extraction and precise partitioning.
  * **Custom ConvVAE Implementation:** End-to-end development of a VAE architecture tailored for image anomaly detection.
  * **Principled Thresholding:** Dynamic, statistically driven threshold calculation for anomaly classification.
  * **Comprehensive Evaluation Pipeline:** Standardized metrics (ROC-AUC, Precision, Recall, F1-Score, Confusion Matrix) applied to both internal and an independent external test set.
  * **Inference Capabilities:** Scripts for applying the trained model to new data and outputting predictions in a structured format.
  * **Visual Diagnostics:** Tools for qualitative analysis of model performance, including reconstructions, error maps, and detailed insights into false positives/negatives.

## 📂 Project Structure

```
.
├── Anomly_detection_concrete.ipynb  # Main project notebook (code, training, evaluation, visualizations)
├── /content/                   # Runtime directory for data & temporary files (Colab context)
│   └── concrete/
│       └── Data/
│           ├── non-crack/      # Remaining 15,000 non-cracked images (for VAE internal train/val/test)
│           ├── crack/          # Remaining 5,000 cracked images (for VAE internal val/test)
│           ├── new_non_crack/  # **Dedicated External Test Set (5,000 non-cracked)**
│           └── new_crack/      # **Dedicated External Test Set (15,000 cracked)**
├── vae_concrete.pth            # Trained VAE model checkpoint
├── non_cracked_concrete_predictions.csv # CSV output of external non-crack predictions
├── cracked_concrete_predictions.csv # CSV output of external crack predictions
├── README.md                   # This README file
└── LICENSE                     # (Optional) Project license
```

-----

## 🔬 Technical Approach

### 1\. Robust Data Preparation

The project utilizes the "Concrete Crack Images" dataset from Kaggle. **Initially, 20,000 images were available for non-cracked concrete and 20,000 images for cracked concrete.** A meticulous data preparation pipeline was implemented to ensure highly reliable model evaluation:

  * **Preprocessing:** Images were converted to grayscale, resized to `128x128`, and normalized.
  * **Strategic Data Partitioning:**
      * **External, Unseen Test Sets:** A significant portion of the original data (5,000 non-crack, 15,000 crack images) was explicitly segregated into `new_non_crack` and `new_crack` folders. This ensures a **truly unbiased final evaluation** that mimics real-world deployment scenarios.
      * **Internal Training & Validation:** The remaining 15,000 non-crack images were split for VAE training (80%), validation (10%), and internal testing (10%). The remaining 5,000 crack images were used for internal validation (10%) and testing (90%) of the anomaly detection capability.
  * **Data Leakage Prevention:** Rigorous checks confirmed **zero overlap** between any of the partitioned datasets, a critical step for valid model assessment.

**Final Dataset Counts (after partition for anomaly detection):**

| Folder Name | Count | Role in Pipeline |
| :---------- | :---- | :--------------- |
| `new_non_crack` | 5000 | **External, Unseen Test Data (Normal)** |
| `new_crack` | 15000 | **External, Unseen Test Data (Anomaly)** |
| `non-crack` | 15000 | Internal VAE Training, Validation, and Test Split |
| `crack` | 5000 | Internal VAE Validation and Test Split (Anomalies) |

### 2\. Custom ConvVAE Architecture

The ConvVAE is engineered to learn the intricate patterns of "normal" concrete surfaces:

  * **Encoder:** Downsamples input through a series of `Conv2d` layers (1 -\> 256 channels), extracting a hierarchical feature representation.
  * **Latent Space:** Maps the encoded features to a 128-dimensional probabilistic latent space (mean and log-variance), from which samples are drawn via the reparameterization trick.
  * **Decoder:** Upsamples the latent sample using `ConvTranspose2d` layers, reconstructing the original `128x128` grayscale image.
  * **Loss Function:** A composite `vae_loss` combining **Mean Squared Error (MSE) for reconstruction accuracy** and **Kullback-Leibler (KL) Divergence** for latent space regularization.

### 3\. Training & Anomaly Detection Pipeline

The VAE was trained for 10 epochs using the Adam optimizer on the **internal `non-crack` training set only** (12,000 images), leveraging GPU acceleration.

**Anomaly Threshold Determination:**
A data-driven threshold is crucial for reliable anomaly detection. The **95th percentile** of reconstruction errors from the internal `non-crack` validation set was chosen as the anomaly threshold.

  * **Calculated Threshold:** `0.011993`

-----

## 📊 Quantifiable Results & Performance

The model's performance was rigorously evaluated on both internal and external test sets. All reported metrics were obtained on the available hardware (CUDA-enabled GPU where available).

### Internal Test Set Performance

These metrics demonstrate the model's ability to discriminate between known normal and anomalous patterns within the initially partitioned data:

| Metric | Value | Interpretation for Anomaly Detection |
| :----- | :---- | :----------------------------------- |
| **ROC-AUC** | **0.9905** | Exceptional ability to distinguish cracked from non-cracked images. |
| **Precision**| **0.9812** | Of all predicted cracks, 98.12% were actual cracks. |
| **Recall** | **0.9649** | The model successfully identified 96.49% of all actual cracks. |
| **F1 Score** | **0.9730** | Strong balance between precision and recall, indicating robust performance. |

-----

**Confusion Matrix (Internal Test Set):**

This matrix provides a detailed breakdown of the classification accuracy on the internal test set:

| Predicted \\ True | NON\_CRACK (0) | CRACK (1) |
| :--------------- | :------------- | :-------- |
| **NON\_CRACK (0)** | 1417 (TN) | 83 (FN) |
| **CRACK (1)** | 158 (FP) | 4342 (TP) |

  * **True Negatives (TN):** 1417 non-cracked images correctly identified as non-cracked.
  * **False Positives (FP):** 158 non-cracked images incorrectly flagged as cracked (False Alarms).
  * **False Negatives (FN):** 83 actual cracked images missed and classified as non-cracked (Critical Misses).
  * **True Positives (TP):** 4342 cracked images correctly identified as cracked.

-----

### Performance on External, Unseen Test Sets

Crucially, the model's robustness was validated against **completely unseen data** to simulate real-world deployment conditions:

  * **Non-Cracked Concrete (5,000 images):** The model accurately classified **4,735 (94.7%)** as non-cracked.
  * **Cracked Concrete (15,000 images):** The model successfully detected **14,475 (96.5%)** as cracked.

These results confirm the model's strong generalization capabilities and its readiness for practical application.

-----

## 📈 Visual Analysis & Interpretability

The project includes comprehensive visualization tools within the notebook to aid in model understanding and debugging:

  * **Reconstruction Examples:** Visual comparison of original images, their VAE reconstructions, and heatmaps of reconstruction errors, providing intuitive insights into anomaly detection.
  * **Correct Prediction Spotlights:** Showcasing instances where the model successfully identified both normal and anomalous cases.
  * **Failure Case Analysis:** Dedicated visualizations for **False Positives** and **False Negatives**, crucial for understanding model limitations and guiding future improvements.

-----

## ✅ Conclusion & Future Outlook

This project built and tested a strong system to find cracks in concrete using a special deep learning model called Convolutional Variational Autoencoder. It shows skills in deep learning, data handling, and model use. In the future, we can try learning on devices, using live video, or improving the model with more training to make it better.


