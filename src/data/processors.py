# import cv2
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
# import albumentations as A
from torchvision import transforms

# Factory functions for transforms (instantiate from cfg)
def get_image_transform(cfg):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.dataset.image_height, cfg.dataset.image_height), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=cfg.dataset.image_transform.mean, std=cfg.dataset.image_transform.std),
    ])

def get_mask_transform(cfg):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.dataset.image_height, cfg.dataset.image_height), interpolation=transforms.InterpolationMode.NEAREST),
    ])

def get_joint_transform(cfg):
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.dataset.joint_transform.horizontal_flip_prob),
            A.VerticalFlip(p=cfg.dataset.joint_transform.vertical_flip_prob),
        ],
        p=1,
    )
