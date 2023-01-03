import streamlit as st
import torch
import torchvision
from torch import nn
import numpy as np
from PIL import Image

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

def preview_Image(image):
    st.image(image, width=540, caption='Your Image')
    st.header("")
    
    


def Input_Image():
    picture_file = st.file_uploader("Upload a image of the airplane")
    
    # st.subheader("OR")
    # picture_camera = st.camera_input("Take a picture!")
    

    if (picture_file != None):
        picture = Image.open(picture_file)
        preview_Image(picture)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(), # 0 -> 255
            torchvision.transforms.Resize((64, 64))
            ])

        X = transform(picture)
        X = X.type(torch.float32)
        X  = X / 255
        X = X.unsqueeze(dim=0)

        print(X.shape)
        print(X.dtype)
        print(X)

        return X
    else:
        return 0
        



def load_model():
    
    model = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=50)
    model.load_state_dict(torch.load('data/Models/models_0.pth'))



    return model


def predict_image(predict):
    path = "data/Aircraft Pictures/" + predict + "/1.jpg"
    st.image(path, width=540 , caption="Prediction")


def show_predictor_page():
    plane_raw = ['A-10A Thunderbolt II',
  'A-37A Dragonfly',
  'A-37A DragonflyAC-130A Spectre',
  'ADM-20 Quail',
  'Airbus 350',
  'Airbus A300',
  'Airbus A380',
  'Airbus Beluga',
  'Airbus BelugaXL',
  'B-17G Flying Fortress',
  'B-1B Lancer',
  'B-29B Superfortress',
  'B-52D Stratofortress',
  'Boeing 777',
  'Boeing 787',
  'C-119C Flying Boxcar',
  'C-123K Provider',
  'C-124C Globemaster II',
  'C-130E Hercules',
  'C-141C Starlifter',
  'C-46D Commando',
  'C-47B Skytrain',
  'C-54G Skymaster',
  'C-7A Caribou',
  'CH-21B Workhorse',
  'EC-121K Constellation',
  'EC-135N Stratotanker',
  'F-100D Super Sabre',
  'F-101F Voodoo',
  'F-102A Delta Dagger',
  'F-105D Thunderchief',
  'F-106A Delta Dart',
  'F-111E Aardvark',
  'F-15A Eagle',
  'F-16A Fighting Falcon',
  'F-4D Phantom II',
  'F-80C Shooting Star',
  'F-84E Thunderjet',
  'F-86H Sabre',
  'F-89J Scorpion',
  'HH-43F Huskie',
  'KC-97L Stratofreighter',
  'MH-53M Pave Low',
  'P-40N Warhawk',
  'P-51H Mustang',
  'SR-71A Blackbird',
  'UC-78B Bamboo Bomber',
  'UH-1P Iroquois',
  'VC-140B JetStar',
  'WB-66D Destroyer']
    

    st.title("Airplane Type Image Detection")
    st.write("""### It can determine whic aircraft is present within the image inputed by the user.""")

    

    X = Input_Image()

    if torch.is_tensor(X):
        model = load_model()
        model.eval()

        with torch.inference_mode():
            image_pred_logits = model(X)

            
            image_pred_probs = torch.softmax(image_pred_logits, dim=1)
            # print(f"Prediction labels: {image_pred_probs}")
            prediction_index = torch.argmax(image_pred_probs)
            # print(f"Prediction index: {prediction_index}")
            airplane_prediction = plane_raw[prediction_index]
            # print(f"Prediction: {airplane_prediction}")
            
            st.subheader("")
            
            st.subheader(f"The airplane in the picture is :green[{airplane_prediction}]")
            
            st.subheader("")

            predict_image(airplane_prediction)


            





    
    
    
    


       
    
    






    
