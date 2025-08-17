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

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

model.save_weights('weights/classifier_model.weights.h5')

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()