# 🌊 Flood Area Segmentation using Deep Learning

This project aims to perform **semantic segmentation** of flooded areas in satellite images using a **pretrained DenseNet121** model as a backbone. The dataset used is provided by [Kaggle - Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

---

## 📁 Dataset

- Contains:
  - `290` RGB satellite images.
  - `290` corresponding binary masks (flooded area = white, non-flooded = black).
- Downloaded via Kaggle API.

---

## 🧪 Project Pipeline

1. **Environment Setup**
   - Uses Google Colab
   - Installs Kaggle API and downloads the dataset

2. **Data Preparation**
   - Images and masks are resized to 256x256
   - Normalization of image pixel values
   - Masks are converted to grayscale
   - Data is converted to TensorFlow `Dataset` objects and split:
     - 85% for training
     - 15% for validation

3. **Model Architecture**
   - Pretrained `DenseNet121` used as encoder (`include_top=False`)
   - Custom decoder:
     - 1x1 Convolution
     - `Conv2DTranspose` for upsampling to original image size
   - Final activation: `sigmoid` for binary segmentation

4. **Training**
   - Optimizer: `Adam` with learning rate `0.001`
   - Loss: `Binary Crossentropy`
   - Metrics: `Accuracy`
   - Callbacks:
     - `ModelCheckpoint` to save the best model
     - `ReduceLROnPlateau` to reduce LR on validation accuracy plateau

---

## 🧠 Model Summary

- Input size: (256, 256, 3)
- Output: Binary mask (256, 256, 1)
- Parameters:
  - Trainable: Yes
  - Pretrained weights from ImageNet used in DenseNet121

---

## 📊 Training Results

- Trained for 50 epochs
- Uses `val_accuracy` to save best-performing model
- Can further visualize training history using `matplotlib`

---

## 🔧 Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- PIL
- Matplotlib
- tqdm
- kaggle

---

## 🚀 How to Run

1. Upload `kaggle.json` to Colab
2. Run the notebook step by step
3. Training will begin and save the best model to `best_model.h5`

---

## 📌 Notes

- You can further improve performance by:
  - Adding data augmentation
  - Using a more advanced decoder (like U-Net or FPN)
  - Experimenting with other pretrained models (ResNet, EfficientNet)

---

## 📬 Contact

For questions or suggestions, feel free to reach out via GitHub Issues.

---

## 📸 Sample Visualization

A few examples showing input images and their corresponding segmentation masks:

| Image | Mask |
|-------|------|
| ![Image](sample1.jpg) | ![Mask](mask1.jpg) |

---

## 📄 License

This project is for educational and research purposes only.
