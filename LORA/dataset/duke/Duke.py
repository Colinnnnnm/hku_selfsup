import os 
import glob
import re
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from collections import defaultdict
from torchvision import transforms
import torch

class Duke_Dataset(Dataset):
    def __init__(self,train_folder,transform=None,mode='L'):
        '''
        Parameter for Duke_dataset
        train_folder: Path to Market1501 data, can be download from https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
        transform: List of torchvision Transformation
        mode: String, Mode that PIL Image support. If "L", the image will be convert to GrayScale. If "RGB", the image will be in RGB scale.
        '''
        self.train_folder =  train_folder
        self.transform = transform
        self.mode=mode
        self._process()
    @staticmethod
    def _process_image_name(name):
        '''
        Image name: {person_id}_c{carema id}s{sequence_number}_{frame number}.jpg
        '''
        pattern = re.compile(r'([-\d]+)_c(\d)')
        return map(int, pattern.search(name).groups())
    def __getitem__(self, pid):
        '''
        Accept pid for the identity
        Return all the image of of it
        '''
        possible = self.dataset[pid]
        output = []
        for img_path, pid_,cid,label in possible:
            img =  self.load_img(img_path)
            output.append((img, label, pid_,cid))
        return output
    def __len__(self):
        return sum([len(x[1]) for x in self.dataset.items()])
            
    def load_img(self,path):
        with open(path,"rb") as f:
            img = Image.open(f).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def _process(self):
        pids = set()
        for img in glob.glob(self.train_folder+"/*.jpg"):
            pid, cid =  self._process_image_name(img)
            if pid!="-1":
                pids.add(pid)
        mapping = {pid:idx for idx,pid in enumerate(pids)} #Make sure the pid to label mapping is monotone
        self.dataset=defaultdict(list)
        for img in glob.glob(self.train_folder+"/*.jpg"):
            pid, cid =  self._process_image_name(img)
            if pid!="-1":
                if pid in self.dataset:
                    self.dataset[mapping[pid]].append((img,pid,cid, mapping[pid]))
                else:
                    self.dataset[mapping[pid]] = [(img,pid,cid, mapping[pid])]