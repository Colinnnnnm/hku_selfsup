from visualizer import get_local
get_local.activate()

from PIL import Image
import torch
from torch import nn
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def visualize_cls(attn_map, imgpath, output_dir, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attn_map_torch = torch.from_numpy(attn_map)
    nh = attn_map_torch.shape[1]
    attn_torch = attn_map_torch[0, :, 0, 1:].reshape(nh, -1)
    #last_attention_map = torch.mean(last_attention_map, dim=0).unsqueeze(0)#separate head, show attention map on different domain and baseline
    attn = attn_torch.numpy()

    imgpath = imgpath[0]
    image_fname = os.path.basename(imgpath)
    image = Image.open(imgpath)

    fig, ax = plt.subplots(1, (1 + nh), figsize=(7, 2))

    ax[0].imshow(image)
    ax[0].axis('off')

    for idx, head in enumerate(attn, start=1):

        mask = head.reshape(grid_size[0], grid_size[1])
        mask = Image.fromarray(mask).resize(image.size)

        mask = mask / np.max(mask)

        ax[idx].imshow(image)
        ax[idx].imshow(mask, alpha=alpha, cmap='rainbow')
        ax[idx].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{image_fname}_attn.png"))

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid.visual", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    # Visualizing Dog Image
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for idx, (img, pid, camid, camids, target_view, imgpath, mapped_img) in enumerate(train_loader_normal):
        if idx >= cfg.DATALOADER.MAX_ROUND:
            break
        get_local.clear()
        img = img.to(device)
        mapped_img = mapped_img.to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        feat = model(img, mapped_x=mapped_img, cam_label=camids, view_label=target_view)

        cache = get_local.cache
        attention_maps = cache['Attention.forward']
        last_attention_map = attention_maps[-1]

        w_featmap, h_featmap = 9, 18

        visualize_cls(last_attention_map, imgpath, output_dir, grid_size=(h_featmap, w_featmap))
