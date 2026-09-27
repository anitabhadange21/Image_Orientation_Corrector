import tensorflow as tf
import os

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("----------------------------------------")
print("Loading training dataset...")
print("----------------------------------------")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


train_data = train_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


test_data = test_datagen.flow_from_directory(
    TEST_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("----------------------------------------")
print("Class labels:")
print(train_data.class_indices)
print("----------------------------------------")


# =========================================================
# BUILD CNN MODEL
# =========================================================

model = models.Sequential([

    # First convolution layer
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third convolution layer
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    # 4 classes = 0, 90, 180, 270
    layers.Dense(
        4,
        activation="softmax"
    )
])


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# DISPLAY MODEL
# =========================================================

print("----------------------------------------")
print("CNN MODEL")
print("----------------------------------------")

model.summary()


# =========================================================
# TRAIN MODEL
# =========================================================

print("----------------------------------------")
print("Training CNN...")
print("----------------------------------------")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)


# =========================================================
# EVALUATE MODEL
# =========================================================

print("----------------------------------------")
print("Evaluating model...")
print("----------------------------------------")

loss, accuracy = model.evaluate(test_data)

print("----------------------------------------")
print("FINAL TEST RESULTS")
print("----------------------------------------")

print("Test Accuracy:", round(accuracy * 100, 2), "%")


# =========================================================
# SAVE MODEL
# =========================================================

model.save("orientation_cnn.keras")

print("----------------------------------------")
print("CNN MODEL SAVED!")
print("File: orientation_cnn.keras")
print("----------------------------------------")