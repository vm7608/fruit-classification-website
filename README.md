# **Fruit Recognition Website**

## **Description**

This is a small project to recognize fruits using 3 popular CNN models: ResNet50, DenseNet121 and MobileNet. These models are trained on our custom dataset of 20 fruits. The dataset is available [here](https://www.kaggle.com/datasets/philosopher0808/dataset-of-20-fruits). The website is built using Streamlit and deployed on Streamlit Cloud. You can access the website [here](https://fruit-classification-website-hmud.streamlit.app/).

## **Installation**

If you want to run this project locally, you can clone this repository and install the required packages using the following command:

```bash
# Clone the repository
git clone https://github.com/vm7608/fruit-classification-website.git

# Create a virtual environment
virtualenv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## **Details of implementation**

We have created a full training/testing pipeline for the models. The pipeline details can be found in the folder `notebooks`. The trained models are saved in the folder `models`.
The code to process data and create the dataset can be found in the folder `data processing`. The code to create the website can be found in the folder `streamlit_app`.

For more details, please refer to the `document` folder.

## **Contributors**

- Cao Kieu Van Manh
- Nguyen Tuan Hung
- Vo Hoang Bao
- Nguyen Tien Hung
