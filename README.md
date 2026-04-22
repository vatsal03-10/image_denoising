# 🧠 Image Denoising using Deep Learning (Autoencoder)

## 📌 Overview

This project implements an **image denoising system using a Convolutional Autoencoder**.
The model is trained on the **CIFAR-10 dataset** to learn how to remove noise from images and reconstruct clean outputs.

---

## 🚀 Features

* 🧠 Deep learning-based denoising (Autoencoder)
* 🖼️ Trained on CIFAR-10 dataset
* 🔊 Artificial noise added to images
* ✨ Reconstructs clean images from noisy input
* 📊 Displays training & validation loss graph
* 📈 Evaluates performance using PSNR & SSIM

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV

---

## 📂 Project Structure

```id="z3n1v8"
Image_Denoising/
 ├── main.py          # Training + testing script
 ├── dexter.jpeg      # Custom input image
 ├── .gitignore
 └── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash id="v0g3qa"
git clone https://github.com/your-username/image_denoising.git
cd image_denoising
```

### 2️⃣ Install Dependencies

```bash id="l7x9mk"
pip install tensorflow numpy matplotlib opencv-python pillow scikit-image
```

---

## ▶️ Usage

Run the script:

```bash id="0c9lfk"
python main.py
```

---

## 🧠 How It Works

1. CIFAR-10 dataset is loaded
2. Images are normalized (0–1 range)
3. Random noise is added to images
4. A **Convolutional Autoencoder** is built:

   * Encoder → extracts features
   * Decoder → reconstructs clean image
5. Model is trained on noisy → clean images
6. Output is evaluated using:

   * PSNR (Peak Signal-to-Noise Ratio)
   * SSIM (Structural Similarity Index)

---

## 🏗️ Model Architecture

* Conv2D layers (feature extraction)
* MaxPooling (downsampling)
* UpSampling (reconstruction)
* Final sigmoid layer for output image

---

## 📊 Output

* Training vs Validation Loss Graph
* Denoised Image Output
* PSNR & SSIM scores printed

---

## 🔮 Future Improvements

* 🔥 Use deeper CNN architecture
* 🤖 Train on custom dataset
* ⚡ Real-time denoising
* 🌐 Deploy as web app
* 🎯 Use GANs for better results

---

## 👨‍💻 Author

**Vatsal Patil**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
