import torch 
from torch import nn

import requests
from pathlib import Path
import os

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision import io

import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np


# Images are broken down to numbers for each pixel that contains three numbers for RGB.

# We shall use a convolutional neural network (CNN) as it provides the best results but we can also use transformers, I may furture use transformers for intergerating different models

# Shape = [batch_size, width, height, colour_channels] (NHWC) 
# Or
# Shape = [batch_size, colour_channels, height, width] (NCHW) 

# What is a convolutional neural netwrok (CNN)?
#  - Input layer -> Convolutional layer -> Hidden activation (linear or non-linear activation) -> Pooling layer -> Output layer -> Ouput activation

# `torchvision.datasets` - get datasets and data loading functions for computer vision
# `torchvision.models` - get pretrained computer vision models that you can leverage for your problens\
# `torchvision.transforms` - functions for manipulating your vision data (images) to be suitable for use with an ML model
# `torch.utils.data.Dataset` - Base dataset class for PyTorch.
# `torch.utils.data.dataLoader` - Creates a Python iterable over a dataset

def dataset_maker(planer_raw):

    PATH = "Aircraft Pictures\\"

    # [image_channels, image_height, image_width, name_aircraft]

    ntrain = int(0.8 * 20)
    ntest = 20 - ntrain

    # Making a train set
    for i in range(len(plane_raw)):
        PATH_PLANE = PATH + "\\" + plane_raw[i]

        for j in range(1, ntrain + 1, 1):
            img = io.read_image(PATH_PLANE + "\\" + str(j) + ".jpg")
            print(img)
        print(img.shape) # [colour_channels, height, width]

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


plane_raw = ["Boeing 787", 
"Airbus A380", 
"Airbus A300", 
"Airbus 350", 
"Airbus Beluga", 
"Airbus BelugaXL", 
"Boeing 777", 
"A-37A Dragonfly", 
"ADM-20 Quail", 
"A-10A Thunderbolt II",
"A-37A Dragonfly"
"AC-130A Spectre", 
"WB-66D Destroyer", 
"B-1B Lancer", 
"B-29B Superfortress", 
"B-52D Stratofortress", 
"B-17G Flying Fortress", 
"C-141C Starlifter", 
"C-7A Caribou", 
"C-47B Skytrain", 
"C-54G Skymaster", 
"C-119C Flying Boxcar",
"C-124C Globemaster II",
"C-130E Hercules",
"EC-135N Stratotanker",
"VC-140B JetStar",
"UC-78B Bamboo Bomber",
"KC-97L Stratofreighter",
"C-46D Commando",
"C-123K Provider",
"EC-121K Constellation",
"F-80C Shooting Star",
"P-40N Warhawk",
"F-84E Thunderjet",
"F-89J Scorpion",
"F-101F Voodoo",
"F-105D Thunderchief",
"F-111E Aardvark",
"F-4D Phantom II",
"F-16A Fighting Falcon",
"F-15A Eagle",
"F-102A Delta Dagger",
"F-106A Delta Dart",
"F-86H Sabre",
"P-51H Mustang",
"F-100D Super Sabre",
"HH-43F Huskie",
"CH-21B Workhorse",
"UH-1P Iroquois",
"MH-53M Pave Low" ,
"SR-71A Blackbird"]

data_path = Path("data/")
image_path = data_path / "dataset"

#walk_through_dir(image_path)

# Set seed
random.seed(42)

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array} -> [height, width, color_channels]")
plt.axis(False)