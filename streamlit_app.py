import io
import os
import uuid

import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from keras.models import Model, load_model
from keras.utils import img_to_array, load_img
from PIL import Image


# Load the model and labels
labels = {
    0: "apple",
    1: "avocado",
    2: "banana",
    3: "cucumber",
    4: "dragonfruit",
    5: "durian",
    6: "grape",
    7: "guava",
    8: "kiwi",
    9: "lemon",
    10: "lychee",
    11: "mango",
    12: "orange",
    13: "papaya",
    14: "pear",
    15: "pineapple",
    16: "pomegranate",
    17: "strawberry",
    18: "tomato",
    19: "watermelon",
}


def resize_image(img_path, size=(224, 224)):
    """This function resize the image to square shape and save it to the same path."""
    img = cv2.imread(img_path)
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
    mask = cv2.resize(mask, size, interpolation)
    cv2.imwrite(img_path, mask)


def fetch_calories(prediction):
    try:
        calories_data = {
            "apple": "52.1",
            "avocado": "160",
            "banana": "89",
            "cucumber": "15",
            "dragonfruit": "60",
            "durian": "149",
            "grape": "69",
            "guava": "68",
            "kiwi": "61",
            "lemon": "29",
            "lychee": "66",
            "mango": "60",
            "orange": "43",
            "papaya": "43",
            "pear": "57",
            "pineapple": "50",
            "pomegranate": "83",
            "strawberry": "32",
            "tomato": "18",
            "watermelon": "30",
        }
        calories = calories_data.get(prediction)
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def get_model_prediction(img_path, model):
    # Load the image
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the image
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    predicted_label = labels[predicted_class]
    return predicted_label, prediction[0][predicted_class] * 100


def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def GradCAM_explain(image_path, model, predicted_class, last_conv_layer=None):
    image = resize_image(image_path)
    preprocessed_image = preprocess_image(image_path)

    # last_conv_layer = "conv2d_13"
    # get name of the last conv layer of the model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(preprocessed_image)
        loss = preds[:, predicted_class]
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    grads *= weights
    grads = tf.reduce_sum(grads, axis=(0, 3))
    grads = tf.nn.relu(grads)
    grads /= tf.reduce_max(grads)
    grads = tf.cast(grads * 255.0, 'uint8')
    cam = np.array(Image.fromarray(grads.numpy()).resize((224, 224)))
    orig_image = image
    cam_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_rgb_resized = cv2.resize(cam_rgb, (orig_image.shape[1], orig_image.shape[0]))
    alpha = 0.5
    overlay = orig_image.copy()
    cv2.addWeighted(cam_rgb_resized, alpha, overlay, 1 - alpha, 0, overlay)
    file_path = save_image_with_uuid(overlay)
    return file_path


def save_image_with_uuid(image):
    # Tạo giá trị hash cho nội dung của file và lưu ảnh
    file_path = f"explain/{str(uuid.uuid4())}.png"
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return file_path


def process_image(img_content, model_name):
    model = load_model(f"models/{model_name}.h5")
    try:
        img = Image.open(io.BytesIO(img_content))
        st.image(img, use_column_width=False, width=700)

        save_image_path = os.path.join('images', f'{str(uuid.uuid4())}.png')
        with open(save_image_path, "wb") as f:
            f.write(img_content)

        resize_image(save_image_path)

        result, percentage = get_model_prediction(save_image_path, model)

        st.success("**Predicted : " + result + '**')
        st.info('**Accuracy : ' + str(round(percentage, 2)) + '%**')
        st.warning(
            '**Calories : ' + fetch_calories(result) + ' calories in 100 grams**'
        )

        st.subheader("**Explain result with GradCAM**")
        try:
            predicted_class = list(labels.values()).index(result)

            if model_name == 'mobilenet':
                last_conv_layer = "conv2d_13"
            elif model_name == 'resnet':
                last_conv_layer = "conv2d_52"
            elif model_name == 'densenet':
                last_conv_layer = "conv2d_119"

            explanation_img_path = GradCAM_explain(
                save_image_path, model, predicted_class, last_conv_layer
            )
            st.image(
                Image.open(explanation_img_path),
                use_column_width=False,
                width=700,
                caption="The red-colored areas represent regions most influential for the complex model’s prediction, with deeper shades of red indicating higher influence",
            )

        except Exception as e:
            st.error("Failed to generate GradCAM explanation. Error: {}".format(str(e)))

    except Exception as e:
        st.error("Can't process the image. Please check the image and try again.")
        print(e)


def process_uploaded_image(img_file, model_name):
    img_content = img_file.read()
    process_image(img_content, model_name)


def process_url_image(img_url, model_name):
    try:
        img_content = requests.get(img_url).content
        process_image(img_content, model_name)

    except Exception as e:
        st.error(
            "Can't process the image from the provided URL. Please check the URL and try again."
        )
        print(e)


def run():
    st.title("Fruit Recognition")
    st.sidebar.title("Choose image source")

    img_option = st.sidebar.radio(" ", ["From PC", "From URL"])

    st.sidebar.title("Choose model to use")
    model_name = st.sidebar.radio(" ", ["MobileNet", "ResNet", "DenseNet"])

    if img_option == "From PC":
        img_file = st.sidebar.file_uploader(
            "Choose an Image", type=["jpg", "png", "webp"]
        )
        if img_file is not None:
            process_uploaded_image(img_file, model_name.lower())

    elif img_option == "From URL":
        img_url = st.sidebar.text_input("Paste Image URL")
        if st.sidebar.button("Process Image"):
            process_url_image(img_url, model_name.lower())


run()
