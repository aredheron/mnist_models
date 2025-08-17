import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,
    Concatenate, LayerNormalization,)
from tensorflow.keras.utils import to_categorical


# Diffusion schedule (cosine scheduling, 500 timesteps)

TIMESTEPS = 500
S = 0.008

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

# Generate schedule
betas = cosine_beta_schedule(TIMESTEPS, S).astype(np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

print(f"Using cosine beta schedule with {TIMESTEPS} timesteps")
print(f"Beta range: {betas.min():.6f} to {betas.max():.6f}")
print(f"Final alpha_cumprod: {alphas_cumprod[-1]:.6f}")

# Model architecture

class SinusoidalEmbedding(tf.keras.layers.Layer):
    """Positional embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def call(self, timesteps):
        half_dim = self.dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(timesteps, tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class TimestepEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.sin_emb = SinusoidalEmbedding(dim)
        self.dense1 = Dense(dim, activation='swish')
        self.dense2 = Dense(dim, activation='swish')
    
    def call(self, timesteps):
        emb = self.sin_emb(timesteps[:, 0])
        emb = self.dense1(emb)
        emb = self.dense2(emb)
        return emb

class ResBlock(tf.keras.layers.Layer):
    """Residual block with time and label conditioning."""
    def __init__(self, channels, time_emb_dim=64):
        super().__init__()
        self.channels = channels
        self.time_emb_dim = time_emb_dim
        
        self.norm1 = LayerNormalization()
        self.conv1 = Conv2D(channels, 3, padding='same')
        
        self.time_proj = Dense(channels)
        self.label_proj = Dense(channels)
        
        self.norm2 = LayerNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same')
        
        self.skip_conv = None
    
    def build(self, input_shape):
        x_shape, time_shape, label_shape = input_shape
        if x_shape[-1] != self.channels:
            self.skip_conv = Conv2D(self.channels, 1, padding='same')
        super().build(input_shape)
    
    def call(self, inputs):
        x, time_emb, label_emb = inputs
        
        h = self.norm1(x)
        h = tf.keras.activations.swish(h)
        h = self.conv1(h)
        
        time_proj = self.time_proj(time_emb)
        label_proj = self.label_proj(label_emb)
        cond = time_proj + label_proj
        cond = cond[:, None, None, :]
        h = h + cond
        
        h = self.norm2(h)
        h = tf.keras.activations.swish(h)
        h = self.conv2(h)
        
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        
        return x + h

def build_diffusion_model():
    """U-Net (~2.4M parameters)"""
    
    img_in = Input(shape=(28, 28, 1), name="x_t")
    lbl_in = Input(shape=(10,), name="y_label") 
    t_in = Input(shape=(1,), name="timestep")
    
    time_emb_dim = 64
    time_embedding = TimestepEmbedding(time_emb_dim)
    time_emb = time_embedding(t_in)
    
    label_emb = Dense(32, activation='swish')(lbl_in)
    label_emb = Dense(time_emb_dim, activation='swish')(label_emb)
    
    # 3-level U-Net
    # Level 0: 28x28
    x0 = Conv2D(48, 3, padding='same')(img_in)
    x0 = ResBlock(48, time_emb_dim)([x0, time_emb, label_emb])
    x0 = ResBlock(48, time_emb_dim)([x0, time_emb, label_emb])
    skip0 = x0
    
    # Level 1: 14x14  
    x1 = MaxPooling2D(2)(x0)
    x1 = ResBlock(96, time_emb_dim)([x1, time_emb, label_emb])
    x1 = ResBlock(96, time_emb_dim)([x1, time_emb, label_emb])
    skip1 = x1
    
    # Level 2: 7x7
    x2 = MaxPooling2D(2)(x1)
    x2 = ResBlock(192, time_emb_dim)([x2, time_emb, label_emb])
    x2 = ResBlock(192, time_emb_dim)([x2, time_emb, label_emb])
    
    # Decoder path
    # Level 2 -> 1
    d1 = UpSampling2D(2, interpolation='bilinear')(x2)
    d1 = Conv2D(96, 3, padding='same')(d1)
    d1 = Concatenate()([d1, skip1])
    d1 = ResBlock(96, time_emb_dim)([d1, time_emb, label_emb])
    d1 = ResBlock(96, time_emb_dim)([d1, time_emb, label_emb])
    
    # Level 1 -> 0
    d0 = UpSampling2D(2, interpolation='bilinear')(d1)
    d0 = Conv2D(48, 3, padding='same')(d0)
    d0 = Concatenate()([d0, skip0])
    d0 = ResBlock(48, time_emb_dim)([d0, time_emb, label_emb])
    d0 = ResBlock(48, time_emb_dim)([d0, time_emb, label_emb])
    
    # Output
    out = LayerNormalization()(d0)
    out = tf.keras.layers.Activation('swish')(out)
    out = Conv2D(1, 3, padding='same', name='pred_noise')(out)
    
    return Model(inputs=[img_in, lbl_in, t_in], outputs=out)

# Build model
model = build_diffusion_model()
model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss="mse")
print(f"Model parameters: {model.count_params():,}")

# Generate noise, train model

def add_noise(x, y):
    B = x.shape[0]
    t = np.random.randint(0, TIMESTEPS, size=B)
    ε = np.random.normal(size=x.shape).astype(np.float32)
    
    a = sqrt_alphas_cumprod[t][:, None, None, None]
    ā = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    x_t = a * x + ā * ε
    
    t_input = t.astype(np.float32).reshape(-1, 1)
    
    return (x_t, y, t_input), ε

def create_training_data(X, y, multiplier=5):
    """Create training dataset"""
    print(f"Creating {multiplier}x training data with on-the-fly noise...")
    
    X_expanded = np.tile(X, (multiplier, 1, 1, 1))
    y_expanded = np.tile(y, (multiplier, 1))
    
    # Generate all the noisy versions
    inputs, targets = add_noise(X_expanded, y_expanded)
    
    return inputs, targets


# Sampling with posterior variance

def sample(digit: int, n: int = 1, use_posterior_variance=True, eta=1.0):
    """Sample a digit"""
    labels = np.tile(to_categorical([digit], 10), (n, 1)).astype(np.float32)
    x = np.random.normal(size=(n, 28, 28, 1)).astype(np.float32)

    # Use fewer sampling steps for cosine schedule
    sampling_timesteps = 50
    step_size = TIMESTEPS // sampling_timesteps
    sampling_indices = list(range(0, TIMESTEPS, step_size))[:sampling_timesteps]
    sampling_indices.reverse()
    
    for i, t in enumerate(sampling_indices):
        if i % 10 == 0:
            print(f"Sampling step {i}/{len(sampling_indices)}")

        t_arr = np.full((n, 1), t, dtype=np.float32)
        ε_hat = model.predict([x, labels, t_arr], verbose=0)

        α_t = alphas[t]
        ᾱ_t = alphas_cumprod[t]
        β_t = betas[t]
        
        if i < len(sampling_indices) - 1:
            t_prev = sampling_indices[i + 1]
            ᾱ_t_prev = alphas_cumprod[t_prev]
        else:
            t_prev = 0
            ᾱ_t_prev = 1.0
        
        pred_x0 = (x - np.sqrt(1 - ᾱ_t) * ε_hat) / np.sqrt(ᾱ_t)
        pred_x0 = np.clip(pred_x0, -1.0, 1.0)
        
        if t_prev > 0:
            dir_xt = np.sqrt(1 - ᾱ_t_prev - eta**2 * β_t) * ε_hat
            noise = np.random.normal(size=x.shape).astype(np.float32)
            x = np.sqrt(ᾱ_t_prev) * pred_x0 + dir_xt + eta * np.sqrt(β_t) * noise
        else:
            x = pred_x0

    return np.clip(x, 0.0, 1.0).reshape(n, 28, 28)

# Mode selection
mode = input("Enter 0 to **train from scratch**, 1 to load weights and **continue** training, 2 to load weights and **skip** training: ").strip()
if mode not in {"0", "1", "2"}:
    raise ValueError("Mode must be '0', '1' or '2'.")

if mode != "0":
    try:
        model.load_weights("weights/diffusion_model.weights.h5", skip_mismatch=True)
        print("Loaded weights from improved_diffusion_model.weights.h5")
    except (OSError, ValueError):
        print("⚠️  No weight file found; starting from random initialization.")

if mode in {"0", "1"}:
    print("\nLoading MNIST...")
    train_df = pd.read_csv("dataset/Train.csv")
    X = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    y = to_categorical(train_df.iloc[:, 0], 10).astype(np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_inputs, train_targets = create_training_data(X_train, y_train, multiplier=2)
    val_inputs, val_targets = create_training_data(X_val, y_val, multiplier=1)
    
    print(f"Training samples: {len(train_targets)}")
    print(f"Validation samples: {len(val_targets)}")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'weights/diffusion_model.weights.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_inputs, train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=15,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!")
else:
    print("Skipping training (mode 2).")

# Generate and save samples
print("\nGenerating samples...")
print("Testing both sampling methods...")

for digit in range(10):
    print(f"\nSampling digit {digit} with DDIM...")
    samples = sample(digit, n=8, use_posterior_variance=True, eta=0.0)
    
    fig, axes = plt.subplots(1, 8, figsize=(6, 3))
    for i in range(8):
        axes[i].imshow(samples[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{digit}")
    
    plt.tight_layout()
    plt.savefig(f"generated_digits/ddim_generated_{digit}.png", dpi=150, bbox_inches="tight")
    plt.show()

print("Sample generation completed!")