from lightly.utils.logger import setup_logger
from lightly.solver.scheduler_factory import create_scheduler
from lightly.processor.simsiam_processor import do_train
import random
import torch
import numpy as np
import os
import argparse

from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.loss import NegativeCosineSimilarity
from lightly.transforms.simsiam_transform import SimSiamTransform
from lightly.models.dinov2.vision_transformer import vit_small_patch14_224_dinov2, vit_base_patch14_224_dinov2

from lightly.models.simsiam import SimSiam

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SimCLR Training")

    parser.add_argument("--model-name", default="transformer", type=str)
    parser.add_argument("--image-height", default=224, type=int)
    parser.add_argument("--image-width", default=224, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--train-batch-size", default=128, type=int)
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--solver-seed", default=1234, type=int)
    parser.add_argument("--output-dir", default="logs/", type=str)
    parser.add_argument("--device-id", default="0", type=str)
    parser.add_argument("--max-epochs", default=120, type=int)
    parser.add_argument("--base-lr", default=0.06, type=float)
    parser.add_argument("--warmup-epochs", default=20, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--log-period", default=10, type=int)
    parser.add_argument("--checkpoint-period", default=60, type=int)
    parser.add_argument("--eval-period", default=5, type=int)
    parser.add_argument("--freeze-cls", default=False, type=bool)
    parser.add_argument("--freeze-pos", default=False, type=bool)
    parser.add_argument("--use-cos-pos", default=False, type=bool)
    parser.add_argument("--freeze-patch", default=False, type=bool)
    parser.add_argument("--freeze-base", default=False, type=bool)
    parser.add_argument("--freeze-base-start", default=6, type=int)
    parser.add_argument("--freeze-base-end", default=-1, type=int)
    parser.add_argument("--pretrain-hw-ratio", default=1, type=int)
    parser.add_argument("--model-fn", default="vit_small_patch14_224_dinov2", type=str)
    parser.add_argument("--train-root",
                        default="datasets/Market-1501-v15.09.15/bounding_box_train",
                        type=str)
    parser.add_argument("--train-mapping-dir",
                        default="datasets/market1501_to_duke/Market-1501-v15.09.15/bounding_box_train",
                        type=str)
    parser.add_argument("--eval-root",
                        default="datasets/Market-1501-v15.09.15/bounding_box_test",
                        type=str)
    parser.add_argument("--eval-mapping-dir",
                        default="datasets/market1501_to_duke/Market-1501-v15.09.15/bounding_box_test",
                        type=str)
    parser.add_argument("--pretrained-path",
                        default="pretrained/dinov2_vits14_pretrain.pth",
                        type=str)
    parser.add_argument("--resume-path",
                        default=None,
                        type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    set_seed(args.solver_seed)

    output_dir = args.output_dir
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("simclr", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    #  logger.info(args)

    logger.info("args: {}".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    input_size = (args.image_height, args.image_width)

    model_dict = {
        "vit_small_patch14_224_dinov2": vit_small_patch14_224_dinov2,
        "vit_base_patch14_224_dinov2": vit_base_patch14_224_dinov2
    }

    backbone = model_dict[args.model_fn](img_size=input_size)
    backbone.load_param(args.pretrained_path, args.pretrain_hw_ratio)
    backbone.freeze(args)

    model = SimSiam(backbone)

    if args.resume_path is not None:
        param_dict = torch.load(args.resume_path, map_location='cpu')
        for k, v in param_dict.items():
            try:
                model.state_dict()[k].copy_(v)
            except:
                pass


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = SimSiamTransform(input_size=input_size)
    # dataset = torchvision.datasets.CIFAR10(
    #     "datasets/cifar10", download=True, transform=transform
    # )
    # or create a dataset from a folder containing images or videos:
    train_dataset = LightlyDataset(args.train_root,
                             args.train_mapping_dir,
                             transform=transform.train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    eval_dataset = LightlyDataset(args.eval_root,
                             args.eval_mapping_dir,
                             transform=transform.eval_transform)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)

    logger.info('===========using cosine learning rate=======')
    scheduler = create_scheduler(args, optimizer)

    do_train(
        args,
        model,
        criterion,
        train_loader,
        eval_loader,
        optimizer,
        scheduler,
        args.local_rank
    )
    #  print(cfg.OUTPUT_DIR)
    #  print(cfg.MODEL.PRETRAIN_PATH)
