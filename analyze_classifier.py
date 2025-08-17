import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical

train_data = pd.read_csv('dataset/Train.csv')
print("Shape of train_data:", train_data.shape)

X = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0]   

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  
X = X.values / 255.0
X = X.reshape(-1, 28, 28, 1)
print("Shape of X after reshaping:", X.shape)

y = to_categorical(y, num_classes=10)
print("Shape of y after one-hot encoding:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(10, (4, 4), activation='gelu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights('weights/classifier_model.weights.h5')

# Get model predictions (probabilities) and convert to class labels
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
y_true_classes = np.argmax(y_val, axis=1)   # Convert one-hot encoded y_val to class indices

# Find misclassified samples
misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
num_misclassified = min(10, len(misclassified_indices))  # Show up to 10 misclassified samples

# Plot the first 10 misclassified images
plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified_indices[:num_misclassified]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_val[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_true_classes[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Print details
print(f"Total misclassified samples in validation set: {len(misclassified_indices)}")
print(f"Showing first {num_misclassified} misclassified samples.")