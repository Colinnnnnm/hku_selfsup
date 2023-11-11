from model.make_model import make_model
from config import cfg
from datasets import make_dataloader
import argparse
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

if args.config_file != "":
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()



cfg.merge_from_file("configs/market/vit_small.yml")
cfg.merge_from_list(args.opts)
cfg.freeze()
train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
