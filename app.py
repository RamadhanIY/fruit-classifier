import streamlit as st
from helper import classify_image
from PIL import Image

class_names = [
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon'
]

model_path = '/kaggle/input/resnet-9/pytorch/default/1/model_state_dict.pth'

st.title('Blur or Bokeh Detector')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    predicted_class,confidence_percentage = classify_image(model_path,image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write(f'Prediction: {predicted_class} ({confidence_percentage})')