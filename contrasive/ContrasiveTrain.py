#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import copy
from info_nce import InfoNCE

class ContrasiveDataset(Dataset):
    def __init__(self,train_folder,image_size = 280):
        self.train_folder =  train_folder
        self.data = glob.glob(train_folder)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.grey_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
        ])
    def __getitem__(self, index):
        '''
        Accept pid for the identity
        Return all the image of of it
        '''
        return self.load_img(self.data[index])
    
    def load_img(self,path):
        with open(path,"rb") as f:
            img = Image.open(f)
            img.load()
            grey= Image.open(f).convert('L')
            grey.load()
        return self.transform(img), self.grey_transform(grey)
            
    def __len__(self):
        return len(self.data)
    
    
def get_train_loader(path, batch_size =64,**kwargs):
    dataset = ContrasiveDataset(path,**kwargs)
    return DataLoader(dataset,batch_size)




def train_constraive_learning(model, train_loader, args):
    teacher = copy.deepcopy(model)
    loss_func = InfoNCE()
    for i,(name, param) in enumerate(teacher.named_parameters()):
            param.requires_grad = False
    
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    running_loss = 0
    for epochs in range(args.epochs):
        for i,(color, grey) in enumerate(train_loader):
            optimizer.zero_grad()

            color_embedding = teacher(color)
            grey_embedding = model(grey)

            loss = loss_func(grey_embedding,color_embedding)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
def main():
    args = parser.parse_args()

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # replace by your saved model
    # Freeze all parameters except the last layer (just for testing purpose)
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < len(list(model.parameters())) - 2:  # Exclude last layer and its bias
            param.requires_grad = False

        
    train_loader = get_train_loader(args.path,args.batch_size)
    # path = "../dataset/market1501/Market1501/bounding_box_train/*.jpg"
    train_constraive_learning(model, train_loader, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrasive Learning Trainer")
    
    parser.add_argument('-p','--path',required = True, help= 'Path to training data')
    # Optimizer
    parser.add_argument('--lr',default = 0.0001, help = 'Learning Rate', type = float)
    parser.add_argument('--weight_decay', default = 5e-4, type = float , help = 'Weight Deacy')
    # Training config
    parser.add_argument('--epochs',default = 1 , type = int)
    parser.add_argument('--batch_size',default= 64, type = int)
    main()





