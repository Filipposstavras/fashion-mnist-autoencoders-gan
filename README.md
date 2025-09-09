# Fashion-MNIST — Autoencoders & GAN (Deep Learning Project)

This project applies **unsupervised deep learning** techniques to the Fashion-MNIST dataset:  
- **Autoencoders** for image compression and reconstruction  
- **Generative Adversarial Networks (GANs)** for synthetic image generation  

Implemented with **TensorFlow/Keras**, including experiments, visualizations, and reproducible reports.

---

## 🔹 Problem 1 — Autoencoders (Reconstruction & Compression)

**Goal:** Learn compressed latent representations of Fashion-MNIST images and reconstruct them.  

- **Dataset:** Fashion-MNIST filtered to 3 classes: *Sandal (5)*, *Sneaker (7)*, *Ankle boot (9)*  
- **Variants:**  
  - **Small Autoencoder:** latent dimension = 32  
  - **Deep Autoencoder:** latent dimension = 64, deeper convolutional layers  
- **Training:** Mean Squared Error loss, Adam optimizer  
- **Evaluation:**  
  - Test reconstruction MSE  
  - Learning curves (loss per epoch)  
  - Side-by-side visualization of original vs reconstructed images  

---

## 🔹 Problem 2 — Generative Adversarial Network (GAN)

**Goal:** Generate synthetic Fashion-MNIST items resembling real samples.  

- **Dataset:** Fashion-MNIST filtered to 3 classes: *T-shirt/top (0)*, *Trouser (1)*, *Pullover (2)*  
- **Preprocessing:** Pixels scaled to [-1, 1] for Tanh generator output  
- **Architecture:**  
  - **Generator:** Dense → Reshape → Conv2DTranspose layers with BatchNorm & LeakyReLU  
  - **Discriminator:** Conv2D layers with LeakyReLU + Dropout → Dense(sigmoid)  
- **Training:** Adversarial training with Binary Crossentropy, Adam(1e-4)  
- **Evaluation:**  
  - Generator vs Discriminator loss curves  
  - Periodic generated sample grids (16 items per epoch checkpoint)  

---

##  How to Run

Install dependencies:
```bash
pip install -r requirements.txt

