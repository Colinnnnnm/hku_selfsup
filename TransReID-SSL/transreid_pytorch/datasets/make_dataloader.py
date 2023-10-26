import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501, Market1501_grey
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .dukemtmcreid import DukeMTMCreID, DukeMTMCreID_grey
from .cuhk03 import CUHK03, CUHK03_grey

__factory = {
    'market1501': Market1501,
    'market1501_grey': Market1501_grey,
    'msmt17': MSMT17,
    'mm': MM,
    'duke': DukeMTMCreID,
    'duke_grey': DukeMTMCreID_grey,
    'cuhk03': CUHK03,
    'cuhk03_grey': CUHK03_grey
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , img_paths, mapped_imgs = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, torch.stack(mapped_imgs, dim=0),

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, mapped_imgs = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, torch.stack(mapped_imgs, dim=0),


# class AspectPad:
#
#     def __init__(self, to):
#         to_h, to_w = to
#         self.to_ratio = to_h / to_w
#
#     def __call__(self, image):
#         w, h = image.size
#         ratio = h / w
#         if ratio == self.to_ratio:
#             return image
#         elif ratio > self.to_ratio:
#             to_w = h / self.to_ratio
#             left_pad = int((to_w - w) / 2)
#             right_pad = int(to_w - left_pad - w)
#             return F.pad(image, [left_pad, 0, right_pad, 0], 0, 'constant')
#         else:
#             to_h = w * self.to_ratio
#             top_pad = int((to_h - h) / 2)
#             bottom_pad = int(to_h - top_pad - h)
#             return F.pad(image, [0, top_pad, 0, bottom_pad], 0, 'constant')

def make_dataloader(cfg):
    train_transforms_list = [
            # AspectPad(cfg.INPUT.SIZE_TRAIN),
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ]

    if cfg.DATALOADER.USE_COLOR_JITTER:
        train_transforms_list.insert(2, T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ))
    if cfg.DATALOADER.USE_GRAYSCALE:
        train_transforms_list.insert(3, T.RandomGrayscale(p=0.2))

    train_transforms = T.Compose(train_transforms_list)

    val_transforms = T.Compose([
        # AspectPad(cfg.INPUT.SIZE_TRAIN),
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms, cfg.DATALOADER.MAPPING_DIR)
    train_set_normal = ImageDataset(dataset.train, val_transforms, cfg.DATALOADER.MAPPING_DIR)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.DATALOADER.MAPPING_DIR)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
