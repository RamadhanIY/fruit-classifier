import torch
import streamlit as st
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class_names = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]


# Define the ResNet9 model
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.rest1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.conv3 = conv_block(64, 64, pool=True)
        self.conv4 = conv_block(64, 128, pool=True)
        self.rest2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.rest1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.rest2(out) + out
        out = self.classifier(out)
        return out


def load_model(model_path):
    model = ResNet9(in_channels=3, num_classes=len(class_names))

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load the modified state dict
    model.load_state_dict(new_state_dict)

    model = model.cpu()
    model.eval()
    return model


def prepare_image(img, image_size=(224, 224)):

    img = img.resize(image_size)
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    normalized_img = (img_tensor - mean) / std
    return normalized_img


def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    denormalized_img = img_tensor * std + mean
    return denormalized_img


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    img_array = tensor.numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def inference(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
    return outputs


def process_output(outputs):
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    predicted_class = class_names[predicted.item()]
    confidence_percentage = confidence.item() * 100
    return predicted_class, confidence_percentage


def classify_image(model_path, img):
    model = load_model(model_path)
    img_tensor = prepare_image(img)
    normalized_img = normalize(img_tensor)
    outputs = inference(model, normalized_img)
    predicted_class, confidence_percentage = process_output(outputs)
    return predicted_class, confidence_percentage


def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


model_path = "model/model_state_dict.pth"

st.title("Fruit Classifier")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    predicted_class, confidence_percentage = classify_image(model_path, image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write(f"Prediction: {predicted_class} ({confidence_percentage})")
