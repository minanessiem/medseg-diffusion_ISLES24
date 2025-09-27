import numpy as np
import torch
from matplotlib import pyplot as plt
from functools import partial

from src.data.processors import get_image_transform, get_mask_transform, get_joint_transform  # Assuming these exist in processors.py; adjust if needed
from src.models.components.diffusion import unnormalize_to_zero_to_one  # For visualization

def test_joint_transform(
    dataset_class,  # e.g., from src.data.loaders import BrainMRIDataset
    df,
    joint_transform,
    mock_joint_transform,  # e.g., A.Compose([], p=1)
    image_transform,
    mask_transform,
    num_samples=5
):
    partial_class = partial(dataset_class, dataframe=df ,image_transform=image_transform, mask_transform=mask_transform)
    org_dataset = partial_class(joint_transform=mock_joint_transform)
    flipped_dataset = partial_class(joint_transform=joint_transform)
    random_indices = df[df['has_cancer'] == 1][['has_cancer']].sample(num_samples).reset_index()['index'].tolist()
    plt.figure(figsize=(10, 4 * num_samples))

    for i, index in enumerate(random_indices):
        org_image, org_mask, _ = org_dataset[index]
        org_image, org_mask = org_image.permute(1, 2, 0).numpy(), org_mask.permute(1, 2, 0).squeeze(2).numpy()

        flipped_image, flipped_mask, _ = flipped_dataset[index]
        flipped_image, flipped_mask = flipped_image.permute(1, 2, 0).numpy(), flipped_mask.permute(1, 2, 0).squeeze(2).numpy()
        org_image[org_mask==1] = (0, 1, .9)
        flipped_image[flipped_mask==1] = (0, 1, .9)
        org_image = unnormalize_to_zero_to_one(org_image)
        flipped_image = unnormalize_to_zero_to_one(flipped_image)
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(org_image)
        plt.title(f"Image")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(flipped_image)
        plt.title(f"Flippd Image")
        plt.axis("off")

    plt.tight_layout()
    plt.show()