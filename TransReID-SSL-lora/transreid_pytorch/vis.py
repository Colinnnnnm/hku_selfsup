from model.make_model import make_model
from config import cfg
from datasets import make_dataloader
cfg.merge_from_file("configs/market/vit_small.yml")

train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
