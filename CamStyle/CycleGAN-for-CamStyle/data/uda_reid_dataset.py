import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os.path as osp
from glob import glob
import re
from data.datasets.cuhk03 import CUHK03
from data.datasets.market1501 import Market1501
from data.datasets.dukemtmcreid import DukeMTMCreID


fun_dict = {
    "cuhk03": CUHK03,
    "market1501": Market1501,
    "dukemtmc": DukeMTMCreID
}

class UDAReidDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.from_dataset = fun_dict[opt.from_name](root=opt.from_path)
        self.to_dataset = fun_dict[opt.to_name](root=opt.to_path)

        self.A_paths = self.preprocess(self.from_dataset.train, cam_id=opt.camA)
        self.B_paths = self.preprocess(self.to_dataset.train, cam_id=opt.camB)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def preprocess(self, paths, cam_id=1, extra_cam_id=-1):
        ret = []
        for path, _, cam, _ in paths:
            if cam not in [cam_id, extra_cam_id]: continue
            ret.append(path)
        return ret

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size) if self.opt.isTrain else self.A_size

    def name(self):
        return 'UnalignedDataset'
