"""Augument images in the dataset by applying random transformations."""

import os
import random
import uuid

import cv2
import numpy as np
import skimage as sk


INPUT_DIR = 'fruit_dataset'
OUTPUT_DIR = 'augumented_fruit_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def flip(img):
    """Flip image horizontally, vertically or both randomly."""
    flip_code = random.choice([-1, 0, 1])
    return cv2.flip(img, flip_code)


def rotate(img):
    """Rotate image randomly between -30 to 30 degrees."""
    angle = random.uniform(-30, 30)
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))


def colorjitter(img, cj_type="b"):
    """Change brightness, contrast or saturation of image randomly."""
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def noisy(image):
    """Add noise to image randomly."""
    noise_typ = random.choice(["gauss", "s&p", "poisson", "speckle"])
    if noise_typ == "gauss":
        noise_image = sk.util.random_noise(image, mode="gaussian")
        noise_image = (255 * noise_image).astype(np.uint8)
        return noise_image
    elif noise_typ == "s&p":
        noise_image = sk.util.random_noise(image, mode="s&p")
        noise_image = (255 * noise_image).astype(np.uint8)
        return noise_image
    elif noise_typ == "poisson":
        noise_image = sk.util.random_noise(image, mode="poisson")
        noise_image = (255 * noise_image).astype(np.uint8)
        return noise_image
    elif noise_typ == "speckle":
        noise_image = sk.util.random_noise(image, mode="speckle")
        noise_image = (255 * noise_image).astype(np.uint8)
        return noise_image


def filters(img):
    """Apply blur, gaussian or median filter to image randomly."""
    f_type = random.choice(["blur", "gaussian", "median"])
    fsize = random.choice([3, 5])
    if f_type == "blur":
        image = img.copy()
        return cv2.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        return cv2.medianBlur(image, fsize)


def run():
    for folder in os.listdir(INPUT_DIR):
        save_dir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(save_dir, exist_ok=True)
        for subfolder in os.listdir(os.path.join(INPUT_DIR, folder)):
            save_subdir = os.path.join(save_dir, subfolder)
            os.makedirs(save_subdir, exist_ok=True)
            for filename in os.listdir(os.path.join(INPUT_DIR, folder, subfolder)):
                img = cv2.imread(os.path.join(INPUT_DIR, folder, subfolder, filename))
                augmented_images = [
                    flip(img),
                    rotate(img),
                    colorjitter(img, cj_type="b"),
                    colorjitter(img, cj_type="s"),
                    colorjitter(img, cj_type="c"),
                    noisy(img),
                    filters(img),
                ]
                for augmented in augmented_images:
                    save_path = os.path.join(save_subdir, str(uuid.uuid4()) + '.jpg')
                    cv2.imwrite(save_path, augmented)


if __name__ == '__main__':
    run()
