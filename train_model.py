import cv2
import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# SETTINGS
# =========================================================

DATASET_FOLDER = "dataset"

IMAGE_SIZE = (128, 128)

ANGLES = [0, 90, 180, 270]


# =========================================================
# HOG FEATURE EXTRACTOR
# =========================================================

hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9
)


def extract_features(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    gray = cv2.resize(gray, IMAGE_SIZE)

    # Extract HOG features
    features = hog.compute(gray)

    # Convert to one-dimensional array
    features = features.flatten()

    return features


# =========================================================
# LOAD DATASET
# =========================================================

print("----------------------------------------")
print("Loading orientation dataset...")
print("----------------------------------------")

X = []
y = []

for angle in ANGLES:

    folder = os.path.join(DATASET_FOLDER, str(angle))

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    files = os.listdir(folder)

    for filename in files:

        file_path = os.path.join(folder, filename)

        features = extract_features(file_path)

        if features is not None:

            X.append(features)
            y.append(angle)


X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))
print("Feature size:", X.shape[1])


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("----------------------------------------")
print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("----------------------------------------")


# =========================================================
# TRAIN SVM CLASSIFIER
# =========================================================

print("Training SVM orientation classifier...")

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# TEST MODEL
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("----------------------------------------")
print("MODEL RESULTS")
print("----------------------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=ANGLES,
        target_names=["0 Degree", "90 Degree", "180 Degree", "270 Degree"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=ANGLES))


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    {
        "model": model,
        "image_size": IMAGE_SIZE
    },
    "orientation_model.pkl"
)

print("----------------------------------------")
print("Model saved successfully!")
print("File created: orientation_model.pkl")
print("----------------------------------------")