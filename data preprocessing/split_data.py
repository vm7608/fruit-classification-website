""" Split data into train, validate and test sets. """""

import os
import random
import shutil

import tqdm


TRAIN_RATIO = 0.6
VALIDATE_RATIO = 0.2
TEST_RATIO = 0.2

source_folder = 'resized_fruit_data'

train_folder = 'fruit_dataset\\train'
validate_folder = 'fruit_dataset\\validate'
test_folder = 'fruit_dataset\\test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(validate_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# ----
# get all subfolders
source_folder = os.listdir(source_folder)
# print(os.listdir(os.path.join('resized_fruit_data', source_folder[0])))
# ----
for i in tqdm.tqdm(range(len(source_folder))):
    images = os.listdir(os.path.join('resized_fruit_data', source_folder[i]))
    num_images = len(images)

    num_train = int(TRAIN_RATIO * num_images)
    num_validate = int(VALIDATE_RATIO * num_images)
    num_test = int(TEST_RATIO * num_images)

    random.shuffle(images)

    train_images = images[:num_train]
    validate_images = images[num_train : num_train + num_validate]
    test_images = images[-num_test:]

    # ----
    train_subfolder_path = os.path.join(train_folder, str(i))
    validate_subfolder_path = os.path.join(validate_folder, str(i))
    test_subfolder_path = os.path.join(test_folder, str(i))
    os.makedirs(train_subfolder_path, exist_ok=True)
    os.makedirs(validate_subfolder_path, exist_ok=True)
    os.makedirs(test_subfolder_path, exist_ok=True)

    # ----
    for img in tqdm.tqdm(
        train_images, desc=f'Copying train images for {source_folder[i]}'
    ):
        shutil.copy(
            os.path.join('resized_fruit_data', source_folder[i], img),
            train_subfolder_path,
        )

    for img in tqdm.tqdm(
        validate_images,
        desc=f'Copying validation images for {source_folder[i]}',
    ):
        shutil.copy(
            os.path.join('resized_fruit_data', source_folder[i], img),
            validate_subfolder_path,
        )

    for img in tqdm.tqdm(
        test_images, desc=f'Copying test images for {source_folder[i]}'
    ):
        shutil.copy(
            os.path.join('resized_fruit_data', source_folder[i], img),
            test_subfolder_path,
        )
