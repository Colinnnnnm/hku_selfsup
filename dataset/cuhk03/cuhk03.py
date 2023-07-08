import os 
import glob
import re
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from collections import defaultdict
from torchvision import transforms
import torch
import pandas as pd
class cuhk03_dateset(Dataset):
    def __init__(self, path, transform=None,mode = 'L'):
        '''
        Parameter for CUHK 03 dataset
        path: Path to CUHK03  data, can be download from https://www.kaggle.com/code/priyanagda/deep-learning-for-re-identification
        transform: List of torchvision Transformation
        mode: String, Mode that PIL Image support. If "L", the image will be convert to GrayScale. If "RGB", the image will be in RGB scale.
        '''
        self.path = path
        self.data = pd.read_csv(path+"/pairs.csv")
        self.image_path = path+"/cuhk03/images_labeled/"
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img1 = Image.open(self.image_path + self.data["image1"][idx]).convert(self.mode)
        img2 = Image.open(self.image_path + self.data["image2"][idx]).convert(self.mode)
        label = self.data["label"][idx]
        
        # Apply image transformations
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label