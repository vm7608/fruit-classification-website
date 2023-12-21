""" Resizes all images in the dataset to 224x224. """""

import os

import cv2
import numpy as np


def resize_image(img, size=(224, 224)):
    """Resize image to square shape of size (size, size) and keep aspect ratio."""
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = (
        cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
    )

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


dataset_path = "fruit_data"

resized_path = "resized_fruit_data"
os.makedirs(resized_path, exist_ok=True)

for folder in os.listdir(dataset_path):
    subfolder = os.path.join(resized_path, folder)
    os.makedirs(subfolder, exist_ok=True)

    for image in os.listdir(os.path.join(dataset_path, folder)):
        filename = image
        try:
            img = cv2.imread(os.path.join(dataset_path, folder, image))
            resized = resize_image(img, (224, 224))
            new_filename = folder + "_" + filename
            cv2.imwrite(os.path.join(subfolder, new_filename), resized)
        except Exception as e:
            print(e)
            print("Error in resizing image: ", filename)
            continue

print('Images resized successfully!')
