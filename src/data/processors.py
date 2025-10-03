# import cv2
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
# import albumentations as A
from torchvision import transforms

def process_and_visualize_cases(mri_df, max_rows, axs):
    """
    Processes and visualizes MRI and mask images for cases with cancer.

    Args:
        mri_df (pd.DataFrame): DataFrame containing MRI image and mask paths, and diagnosis.
        max_rows (int): Maximum number of rows to display in the visualization.
        axs (np.ndarray): Array of matplotlib Axes for displaying the images.

    Returns:
        int: Number of rows visualized.
    """
    row_count = 0

    for idx in mri_df.index:
        try:
            # Check if the current case has cancer
            has_cancer = mri_df.at[idx, 'has_cancer']
            if has_cancer == 1:
                # Load the MRI image and mask
                img_path = mri_df.at[idx, 'image_path']
                mask_path = mri_df.at[idx, 'mask_path']
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Adjust if needed
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Visualize the images
                visualize_img(axs[row_count][0], img, 'Brain MRI')  # From utils/general
                visualize_img(axs[row_count][1], mask, 'Mask', is_mask=True)

                # Overlay mask on the MRI image
                overlayed_img = img.copy()
                overlayed_img[mask == 255] = (0, 255, 150)
                visualize_img(axs[row_count][2], overlayed_img, 'MRI with Mask')

                # Increment the row count
                row_count += 1

            # Stop when maximum rows are filled
            if row_count == max_rows:
                break

        except Exception as e:
            print(f"Error visualizing row {idx}: {e}")

    fig.tight_layout()
    plt.show()

def display_cancerous_rate(mri_df):
    """
    Displays the distribution of cancerous and non-cancerous cases in the dataset.

    Args:
        mri_df (pd.DataFrame): DataFrame containing MRI image and mask paths, and diagnosis.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x='has_cancer', data=mri_df, palette=['#3498db', '#e74c3c'])
    plt.title('Distribution of Cancerous vs Non-Cancerous Cases', fontsize=16)
    plt.xlabel('Diagnosis', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    ax.set_xticklabels(['Non-Cancerous', 'Cancerous'], fontsize=12)
    ax.set_ylim(0, 3000)  # Set y-axis height to 3000

    # Calculate percentages
    total = len(mri_df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=18, color='black', fontweight='bold')

    plt.show()

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
