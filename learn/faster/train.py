from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
from mmdet.apis import set_random_seed

cfg = Config.fromfile('../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg.optimizer.lr = 0.02 / 8
cfg.work_dir = '/media/sata/public-data/result/mmdetection/learn/'
cfg.seed = 1
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
# print(cfg.pretty_text)

dataset = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model, train_cfg=cfg.train_cfg,
                       test_cfg=cfg.test_cfg)

train_detector(model, dataset, cfg, distributed=False, validate=True)