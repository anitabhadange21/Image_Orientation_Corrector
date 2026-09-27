import os
import shutil
import random

# Original dataset
source_folder = "dataset"

# New folders
train_folder = "train"
test_folder = "test"

# Four orientation classes
angles = ["0", "90", "180", "270"]

# Make folders
for angle in angles:
    os.makedirs(os.path.join(train_folder, angle), exist_ok=True)
    os.makedirs(os.path.join(test_folder, angle), exist_ok=True)

# Get original image names
original_images = []

for filename in os.listdir(os.path.join(source_folder, "0")):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        original_images.append(filename)

# Shuffle images
random.seed(42)
random.shuffle(original_images)

# 80% training, 20% testing
split_point = int(len(original_images) * 0.8)

train_images = original_images[:split_point]
test_images = original_images[split_point:]

print("Total original images:", len(original_images))
print("Training originals:", len(train_images))
print("Testing originals:", len(test_images))

# Copy all four orientations
for filename in train_images:

    for angle in angles:

        source = os.path.join(
            source_folder,
            angle,
            filename
        )

        destination = os.path.join(
            train_folder,
            angle,
            filename
        )

        shutil.copy2(source, destination)


for filename in test_images:

    for angle in angles:

        source = os.path.join(
            source_folder,
            angle,
            filename
        )

        destination = os.path.join(
            test_folder,
            angle,
            filename
        )

        shutil.copy2(source, destination)


print("--------------------------------")
print("Dataset split completed!")
print("--------------------------------")
print("Training images:", len(train_images) * 4)
print("Testing images:", len(test_images) * 4)
print("--------------------------------")