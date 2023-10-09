import glob
import re

import os.path as osp

from data.datasets.bases import BaseImageDataset
from collections import defaultdict
import pickle

import json
import os
import errno
import logging

logger = logging.getLogger("camstyle")


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

class CUHK03(BaseImageDataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_
    
    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03'

    def __init__(
            self,
            root='',
            split_id=0,
            cuhk03_labeled=False,
            verbose=True,
            pid_begin=0,
            **kwargs
    ):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json'
        )
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json'
        )

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat'
        )
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat'
        )

        self._check_before_run()
        self.pid_begin = pid_begin

        if verbose:
            logger.info("=> CUHK03 loaded")

        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_new_lab_json_path
        else:
            split_path = self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        self.train = split['train']
        self.query = split['query']
        self.gallery = split['gallery']

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not osp.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))

    def preprocess_split(self):
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
           and osp.exists(self.imgs_detected_dir) \
           and osp.exists(self.split_new_det_json_path) \
           and osp.exists(self.split_new_lab_json_path):
            return

        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2])# - 1 # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, self.pid_begin + int(pid), camid, 1))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1 # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        logger.info('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        logger.info('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)


class CUHK03_grey(BaseImageDataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_

    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03_grey'

    def __init__(
            self,
            root='',
            split_id=0,
            cuhk03_labeled=False,
            verbose=True,
            pid_begin=0,
            **kwargs
    ):
        super(CUHK03_grey, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json'
        )
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json'
        )

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat'
        )
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat'
        )

        self._check_before_run()
        self.pid_begin = pid_begin

        if verbose:
            logger.info("=> CUHK03 loaded")

        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_new_lab_json_path
        else:
            split_path = self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        self.train = split['train']
        self.query = split['query']
        self.gallery = split['gallery']

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not osp.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))

    def preprocess_split(self):
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
                and osp.exists(self.imgs_detected_dir) \
                and osp.exists(self.split_new_det_json_path) \
                and osp.exists(self.split_new_lab_json_path):
            return

        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2]) - 1  # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, self.pid_begin + int(pid), camid, 1))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        logger.info('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        logger.info('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)