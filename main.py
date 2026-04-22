import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from PIL import Image

# 1. TRAIN MODEL
print("Training model...")

(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Add noise
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Build model
input_img = layers.Input(shape=(32, 32, 3))

x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, padding='same')(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)

decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

model = models.Model(input_img, decoded)
model.compile(optimizer='adam', loss='mse')

# Train
history=model.fit(x_train_noisy, x_train,
          epochs=5,
          batch_size=128,
          validation_data=(x_test_noisy, x_test))

print("Model training done!")

# LOSS GRAPH
plt.figure()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()



# 2. TAKE IMAGE INPUT
print("\nSelect an image to denoise...")
file_path = input("Enter image path (drag & drop works): ").strip().replace("'", "")

# Load image
img = Image.open(file_path).convert("RGB")
img = img.resize((32, 32))

# Convert to array
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Add noise (demo)
noise_factor = 0.2
noisy_img = img_array + noise_factor * np.random.normal(size=img_array.shape)
noisy_img = np.clip(noisy_img, 0., 1.)

# 3. DENOISE
denoised_img = model.predict(noisy_img)

# EVALUATION 
img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
denoised_img = tf.convert_to_tensor(denoised_img, dtype=tf.float32)

psnr = tf.image.psnr(img_array, denoised_img, max_val=1.0)
ssim = tf.image.ssim(img_array, denoised_img, max_val=1.0)

print("PSNR:", float(np.mean(psnr)))
print("SSIM:", float(np.mean(ssim)))

# 4. SHOW RESULT
plt.figure(figsize=(8,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Noisy")
plt.imshow(noisy_img[0])
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Denoised")
plt.imshow(denoised_img[0])
plt.axis('off')

plt.suptitle(f"PSNR: {float(np.mean(psnr)):.2f} | SSIM: {float(np.mean(ssim)):.2f}")

plt.show()