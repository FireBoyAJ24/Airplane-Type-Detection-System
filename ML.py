import torch 
from torch import nn

import requests
from pathlib import Path
import os

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchinfo
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

data_transform = transforms.Compose([
    # Resizing the image to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probablility pf flip
    transforms.RandomVerticalFlip(p=0.5),
    # Turn the image inot a torch.Tensor
    transforms.ToTensor() # Converts all pixel values from 0 to 255 to be between 0.0 to 1.0
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)

train_dir = "data\\dataset\\train"
test_dir = "data\\dataset\\test"

train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

print(f"Class names in train_data: {train_data.classes}. Class names in test_data: {test_data.classes}")

print(f"Dictionary: {train_data.class_to_idx}")


class_names = train_data.classes
class_dict = train_data.class_to_idx

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")
print("\n")

# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14);

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, 
                             shuffle=False) # don't usually need to shuffle testing data


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

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)) #.to(device)
print(model_0)

# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(torch.device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")

"""
train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

# Create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes))
print(model_1)

# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_1
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
"""