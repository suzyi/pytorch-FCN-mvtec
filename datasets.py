import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F
import cv2
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(size=(160, 160)), 
    transforms.ToTensor(), 
    ])

def onehot(data):
    """
    inputs:
        data (torch.tensor): shape=(1, height, width)

    outputs:
        label (torch.tensor): shape=(2, height, width), 
    """
    channels, height, width = data.shape
    data = data.int()
    label = torch.zeros(size=(2, height, width), dtype=torch.int)
    label[0:1, :, :] = data
    label[1:2, :, :] = 1 - label[0, :, :]
    label = label.float()
    return label

class mvtecDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.imgs = []
        self.imgs_mask = []
        for name in os.listdir("data/images/"):
            if name[-4:]==".png":
                self.imgs.append(os.path.join("data/images/", name))
                No, Png = name.split(".")
                self.imgs_mask.append(os.path.join("data/ground_truth/", f"{No}_mask.png"))

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img_mask = Image.open(self.imgs_mask[idx])
        if self.transform:
            img = self.transform(img)
            img_mask = self.transform(img_mask)
        img_mask = onehot(img_mask)
        return img, img_mask
        
    def __len__(self):
        return len(self.imgs)



dataset_all = mvtecDataset(transform)

test_size = 2
train_size = len(dataset_all) - test_size

train_dataset, test_dataset = random_split(dataset_all, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, num_workers=2)