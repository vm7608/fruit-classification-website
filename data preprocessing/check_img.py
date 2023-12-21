""" This script checks if all images in the dataset are of size 224x224x3. """

import os

import cv2


def check_image_size(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image:", image_path)
        return False

    height, width, channels = img.shape
    if height == 224 and width == 224 and channels == 3:
        return True
    else:
        return False


dataset_path = "resized_fruit_data"

for folder in os.listdir(dataset_path):
    for image in os.listdir(os.path.join(dataset_path, folder)):
        filename = image
        try:
            if check_image_size(os.path.join(dataset_path, folder, image)):
                continue
            else:
                print("Image size is not 224x224x3:", filename)
        except Exception as e:
            print(e)
            print("Error in resizing image: ", filename)
            continue

print('Images resized successfully!')
