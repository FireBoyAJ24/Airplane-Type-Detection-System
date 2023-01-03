import zipfile
import requests
import shutil
from pathlib import Path

import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

import torch
from torch import nn
from torchinfo import summary

from tqdm.auto import tqdm

from timeit import default_timer as timer

data_path = Path("data/")
image_path = data_path / "Aircraft Pictures"

if (not image_path.is_dir()):
  print(f"There is no {image_path} directory, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

  """
  with open(data_path / "Pics.zip", "wb") as f:
    request = requests.get("https://github.com/FireBoyAJ24/Airplane-Type-Detection-System/raw/Collecting-Images/data/Pics.zip")
    print("Downloading Airplane images files")
    
    f.write(request.content)
  """

  with zipfile.ZipFile(data_path / "Pics.zip", "r") as zip_ref:
    print("Unzipping Aircraft pictures...")
    zip_ref.extractall(image_path)
  

target_directory = "data/Aircraft Pictures"


class_found_names = list()

for entry in list(os.scandir(target_directory)):
  class_found_names.append(entry.name)

class_names = sorted(class_found_names)

class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

def make_standard_image_dir(class_names, num_images):
  # Make a directory called database
  data_path = Path("data/")
  main_path = data_path / "database"
  train_path = main_path / "train"
  test_path = main_path / "test"

  if (main_path.is_dir()):
    print("The main directory", main_path, "Exists")
  else:
    main_path.mkdir(parents=True)

    train_path.mkdir(parents=True)
    test_path.mkdir(parents=True)
  
  """
  1. Scan Aircraft Pictures/Aircraft names
  2. Make a new path for each aircraft names in train and test
  3. Copy the first 10 images to train and then the last 10 images to test
  """
  source_directory = Path("data/Aircraft Pictures/")
  
  print("Standardising the data....")

  # Copying images into a standard image directory organisation
  for p in range(len(class_names)):
    class_directory = source_directory / class_names[p]
    
    for n in range(1, num_images + 1):
      image_path = (str(n) + ".jpg")
      image_source_path = class_directory.joinpath(image_path)
      des_test_class_directory = test_path / class_names[p]
      des_train_class_directory = train_path / class_names[p]
      
      if (image_source_path.is_file()):
        
        if (n < int(3 * num_images/4)):
            
            if (des_train_class_directory.is_dir() != True):         
                des_train_class_directory.mkdir(parents=True)
            
            plane_desc_path = train_path.joinpath(class_names[p])
            image_desc_path = plane_desc_path.joinpath(image_path)
            
            shutil.copy(image_source_path, image_desc_path)
        else:

            if (des_test_class_directory.is_dir() != True):         
                des_test_class_directory.mkdir(parents=True)
            
            plane_desc_path = test_path.joinpath(class_names[p])
            image_desc_path = plane_desc_path.joinpath(image_path)
            
            shutil.copy(image_source_path, image_desc_path)


# make_standard_image_dir(class_names, 20)


main_path = Path("data/database")
train_path = main_path / "train"
test_path = main_path / "test"

data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Flip the images randomly on the vertical
    transforms.RandomVerticalFlip(p=0.5),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

train_data = datasets.ImageFolder(root=train_path,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_path,
                                  transform=data_transform,
                                  target_transform=None)

print(f"Train data:\n {train_data}\nTest data:\n{test_data}")


BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

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

torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes))

summary(model_1, input_size=[1, 3, 64, 64])


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
  model.eval()

  test_loss, test_acc = 0, 0

  with torch.inference_mode():

    for batch, (X, y) in enumerate(dataloader):

      test_pred_logits = model(X)

      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y)).sum().item()/len(test_pred_labels)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
  
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
  }
  
  for epoch in tqdm(range(epochs)):

    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)
    
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn)
    
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )
  
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results

torch.manual_seed(42)

NUM_EPOCHS = 100

model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)


start_time = timer()

model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")