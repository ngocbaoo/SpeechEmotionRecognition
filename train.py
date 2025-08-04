import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

# Load preprocessed features and labels
X = np.load("preprocessed/X_balanced.npy")          # Shape: (num_samples, 40, 862)
y = np.load("preprocessed/y_balanced.npy")          # Shape: (num_samples,) - string labels like 'happy', 'sad'

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X for CNN input (add channel dimension)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Shape: (num_samples, 40, 862, 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)

# Build the model
model = build_model(input_shape=(40, 862, 1), num_classes=y_categorical.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop],
    verbose=1
)
