# Colorectal Polyp Segmentation using Multi-Task Deep Learning

## Introduction
This project focuses on the automated segmentation of colorectal polyps from colonoscopy images using deep learning. Accurate segmentation of polyps is an essential step in early detection and prevention of colorectal cancer. The work proposes an end-to-end multi-task learning framework that combines convolutional neural networks and transformer-based attention mechanisms to achieve robust and precise pixel-level segmentation.

The primary objective of the model is segmentation, while an auxiliary classification task is incorporated to improve global semantic understanding and reduce false positives.

---

## Objectives
- Perform pixel-wise segmentation of colorectal polyps from colonoscopy images  
- Improve segmentation robustness using multi-task learning  
- Capture global contextual information through transformer-based attention  
- Enhance boundary precision using boundary-aware loss functions  
- Compare performance with commonly used baseline models  

---

## Datasets
The model is trained and evaluated using publicly available colorectal polyp datasets:

- Kvasir-SEG  
- CVC-ClinicDB  

Due to size and licensing restrictions, the datasets are not included in this repository.

---

## Model Architecture
The proposed architecture is a unified, end-to-end multi-task network consisting of the following components:

- **ConvNeXt Encoder**  
  A modern convolutional backbone used for hierarchical feature extraction.

- **Transformer Bottleneck**  
  Introduced to model long-range dependencies and capture global context within the image.

- **U-Net Style Decoder**  
  Responsible for restoring spatial resolution and generating pixel-level segmentation masks.

- **Multi-Task Learning Heads**
  - Segmentation head (primary task)
  - Classification head (auxiliary task used for regularization)

The classification head predicts the presence of a polyp at the image level and is not intended for object detection or bounding box prediction.

---

## Training Details
- Framework: TensorFlow / Keras  
- Platform: Google Colab  
- Optimization: AdamW optimizer with learning rate scheduling  
- Loss Function:
  - Binary Cross-Entropy
  - Dice Loss
  - Boundary Loss (for contour refinement)
- Evaluation Metrics:
  - Dice Coefficient
  - Intersection over Union (IoU)

---

## Results
The final model achieves the following performance on the test set:

| Metric | Value |
|------|------|
| Dice Coefficient | ~0.66 |
| Intersection over Union (IoU) | ~0.51 |

Boundary-aware fine-tuning improves contour accuracy and reduces over-segmentation, particularly in challenging cases with weak boundaries.

---

## Qualitative Evaluation
The model produces high-resolution segmentation masks that closely match ground-truth annotations. Qualitative results include visual comparisons of:
- Input colonoscopy images  
- Ground-truth masks  
- Predicted segmentation masks  

These visualizations demonstrate the model’s ability to localize and segment polyps accurately across different image conditions.

---

## Comparison with Baseline Models
The proposed model is compared with previously implemented baseline approaches:

- MobileNet-based U-Net (lightweight baseline)  
- YOLOv8 + U-Net++ (detection-guided segmentation)

While detection-guided approaches may achieve higher raw Dice scores due to region-of-interest filtering, the proposed model offers a fully end-to-end alternative with stronger global context modeling and architectural simplicity.

---

## How to Run
1. Open the provided notebook in Google Colab  
2. Upload the datasets to Google Drive  
3. Update dataset paths in the notebook as required  
4. Run the notebook cells sequentially  

The notebook contains the complete pipeline, including data preparation, training, evaluation, and visualization.

---

## Repository Structure
├── colorectal_polyp_segmentation.ipynb
├── README.md
├── requirements.txt


---

## Notes
- Trained model weights are not included due to file size limitations  
- This repository is intended for academic and research purposes  

---

## Author
AK

---

## License
This project is intended for educational and research use only.
