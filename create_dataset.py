import cv2
import os
import glob

# Folders
input_folder = "original_images"
output_folder = "dataset"

# Create output folders
for angle in [0, 90, 180, 270]:
    os.makedirs(os.path.join(output_folder, str(angle)), exist_ok=True)

# Get all images
image_files = glob.glob(os.path.join(input_folder, "*"))

count = 0

for file in image_files:

    image = cv2.imread(file)

    if image is None:
        continue

    filename = os.path.basename(file)

    # 0 degree
    cv2.imwrite(
        os.path.join(output_folder, "0", filename),
        image
    )

    # 90 degree clockwise
    rotated_90 = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

    cv2.imwrite(
        os.path.join(output_folder, "90", filename),
        rotated_90
    )

    # 180 degree
    rotated_180 = cv2.rotate(
        image,
        cv2.ROTATE_180
    )

    cv2.imwrite(
        os.path.join(output_folder, "180", filename),
        rotated_180
    )

    # 270 degree clockwise
    rotated_270 = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    cv2.imwrite(
        os.path.join(output_folder, "270", filename),
        rotated_270
    )

    count += 1

print("Images processed:", count)
print("Dataset created successfully!")
print("Total images created:", count * 4)
