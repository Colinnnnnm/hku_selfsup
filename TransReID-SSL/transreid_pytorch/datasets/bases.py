import os.path

from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import logging

from torchvision.transforms import InterpolationMode
from timm.data.random_erasing import RandomErasing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset,
                 transform="train"):
        self.cfg = cfg
        self.dataset = dataset
        self.transform = transform
        self.base_transforms_list = [
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            ] if self.transform == "train" else [
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tranforms_list = self.base_transforms_list
        mapping_transforms_list = self.base_transforms_list
        if self.transform == "train":
            train_insert_list = []
            mapping_insert_list = []
            if self.cfg.DATALOADER.USE_COLOR_JITTER:
                train_insert_list.append(T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ))
            if self.cfg.DATALOADER.USE_GRAYSCALE:
                gray_random = torch.rand(1)
                if gray_random < 0.2:
                    def grayscale(img):
                        num_output_channels = F.get_image_num_channels(img)
                        return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
                    train_insert_list.append(grayscale)
                    if self.cfg.DATALOADER.MAPPING_GRAYSCALE:
                        mapping_insert_list.append(grayscale)

            tranforms_list = self.base_transforms_list[:1] + train_insert_list + self.base_transforms_list[1:]
            mapping_transforms_list = self.base_transforms_list[:1] + mapping_insert_list + self.base_transforms_list[1:]

        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        img = T.Compose(tranforms_list)(img)

        mapped_img = img
        if self.transform == "train" and self.cfg.SOLVER.USE_CONTRASTIVE:
            mapped_path = img_path
            if self.cfg.DATALOADER.MAPPING_DIR:
                image_name = os.path.basename(mapped_path)
                folder_name = os.path.basename(os.path.dirname(mapped_path))
                mapped_path = os.path.join(self.cfg.DATALOADER.MAPPING_DIR, folder_name, image_name)
            mapped_img = read_image(mapped_path)
            mapped_img = T.Compose(mapping_transforms_list)(mapped_img)

        return img, pid, camid, trackid, img_path, mapped_img
        #  return img, pid, camid, trackid,img_path.split('/')[-1]
