import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv2DTranspose,
    Conv2D, BatchNormalization, Flatten, LeakyReLU)
from tensorflow.keras.utils import to_categorical

# Build models

generator = Sequential([
    Input(shape=(100,)),
    Dense(7*7*256, activation='relu'),
    Reshape((7, 7, 256)),
    Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'), 
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'),
])

discriminator = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(64, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(0.2),
    Conv2D(128, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2D(256, kernel_size=4, strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    Flatten(),
    Dense(1, activation='sigmoid'),
])

print(f"Generator parameters: {generator.count_params():,}")
print(f"Discriminator parameters: {discriminator.count_params():,}")

# Load and preprocess data
labeled_digits = pd.read_csv('dataset/Train.csv')

X = labeled_digits.iloc[:, 1:]

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  
X = X.values / 255.0
X = (X - 0.5) / 0.5  # Normalize to [-1, 1] for tanh output
X = X.reshape(-1, 28, 28, 1)
print("Shape of X after reshaping:", X.shape)

# Training setup
BATCH_SIZE = 64
NOISE_DIM = 100
EPOCHS = 100

# Optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)  # Slower for discriminator

# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Training step functions
@tf.function
def train_discriminator(real_images):
    batch_size = tf.shape(real_images)[0]
    
    # Generate fake images
    noise = tf.random.normal([batch_size, NOISE_DIM])
    fake_images = generator(noise, training=False)
    
    with tf.GradientTape() as tape:
        # Get discriminator predictions
        real_pred = discriminator(real_images, training=True)
        fake_pred = discriminator(fake_images, training=True)
        
        # Calculate losses
        real_loss = cross_entropy(tf.ones_like(real_pred), real_pred)
        fake_loss = cross_entropy(tf.zeros_like(fake_pred), fake_pred)
        total_loss = real_loss + fake_loss
    
    # Update discriminator
    gradients = tape.gradient(total_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    return total_loss

@tf.function
def train_generator(batch_size):
    noise = tf.random.normal([batch_size, NOISE_DIM])
    
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        fake_pred = discriminator(fake_images, training=False)
        
        # Generator wants discriminator to think fake images are real
        gen_loss = cross_entropy(tf.ones_like(fake_pred), fake_pred)
    
    # Update generator
    gradients = tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    return gen_loss

# Function to generate and display sample images
def generate_and_save_images(epoch):
    noise = tf.random.normal([16, NOISE_DIM])
    generated_images = generator(noise, training=False)
    
    # Convert from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'gan_epoch_{epoch}.png')
    plt.close()

# Training loop
print("Starting GAN training...")

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    disc_losses = []
    gen_losses = []
    
    for batch in dataset:
        # Train discriminator
        disc_loss = train_discriminator(batch)
        disc_losses.append(disc_loss)
        
        # Train generator
        for _ in range(2):
            gen_loss = train_generator(BATCH_SIZE)
            gen_losses.append(gen_loss)
    
    # Print progress
    avg_disc_loss = np.mean(disc_losses)
    avg_gen_loss = np.mean(gen_losses)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - D_loss: {avg_disc_loss:.4f}, G_loss: {avg_gen_loss:.4f}")
    
    generate_and_save_images(epoch + 1)

print("Training completed!")

# Save model weights
generator.save_weights('weights/gan_generator.weights.h5')
discriminator.save_weights('weights/gan_discriminator.weights.h5')

print("Model weights saved to gan_generator.weights.h5 and gan_discriminator.weights.h5")

# Generate final sample
generate_and_save_images('final')
print("Final sample saved as gan_epoch_final.png")