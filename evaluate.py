import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model("best_model.h5")

# Load preprocessed features and labels
X = np.load("X_balanced.npy")
y = np.load("y_balanced.npy")

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X for CNN input (add channel dimension)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Split into training and validation sets (using the same split as in training)
_, X_test, _, y_test_cat = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)
_, _, _, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)


# Make predictions
y_pred_cat = model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# --- Evaluation ---

# 1. Accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 2. Classification Report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()