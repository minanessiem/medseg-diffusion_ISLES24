import torch
import torch.nn as nn
from medpy import metric
import numpy as np
import cc3d
from scipy.ndimage import distance_transform_edt, binary_erosion
from monai.metrics import compute_hausdorff_distance, compute_surface_dice
from monai.networks.utils import one_hot


class DiceMedpyCoefficient(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        # Registered buffers are created on CPU by default.
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Disable gradient tracking for evaluation metrics.
        with torch.no_grad():
            # Immediately detach and move inputs to CPU.
            y_pred_cpu = y_pred.detach().cpu()
            y_true_cpu = y_true.detach().cpu()
            
            # Binarize predictions and ground truth.
            y_pred_binary = (y_pred_cpu > 0.5).float()
            y_true_binary = (y_true_cpu > 0.5).float()

            # Compute intersection and cardinalities on CPU.
            intersection = torch.sum(y_pred_binary * y_true_binary)
            cardinality_pred = torch.sum(y_pred_binary)
            cardinality_true = torch.sum(y_true_binary)

            # Compute Dice coefficient.
            dice = (2.0 * intersection + self.epsilon) / (cardinality_pred + cardinality_true + self.epsilon)

            # Update running sum and count (both kept on CPU).
            self.sum += dice
            self.count += 1

            # Return the dice coefficient computed on CPU.
            return dice

    def compute(self):
        # Compute the average Dice coefficient over all batches.
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach().cpu()
        else:
            return torch.tensor(0.0, device='cpu')

    def reset(self):
        # Reset the running sums and counts.
        self.sum.zero_()
        self.count.zero_()


class DiceNativeCoefficient(nn.Module):
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true, debug=False):
        # Ensure inputs are on the same device
        y_pred = y_pred.to(y_true.device)

        # Binarize predictions and ground truth
        y_pred_binary = (y_pred > self.threshold).float()
        y_true_binary = (y_true > 0).float()

        # Compute sum of both tensors
        pred_sum = torch.sum(y_pred_binary)
        true_sum = torch.sum(y_true_binary)

        if debug:
            print(f"Debug DiceEZCoefficient:")
            print(f"  Original y_pred sum: {torch.sum(y_pred)}")
            print(f"  Original y_true sum: {torch.sum(y_true)}")
            print(f"  Binarized y_pred sum: {pred_sum}")
            print(f"  Binarized y_true sum: {true_sum}")

        if pred_sum == 0 and true_sum == 0:
            dc = 0  # Both are empty, return 0
            if debug:
                print("  Both pred and gt are empty")
        else:
            # Compute intersection
            intersection = torch.sum(y_pred_binary * y_true_binary)
    
            # Compute Dice coefficient
            dc = (2.0 * intersection) / (pred_sum + true_sum + 1e-8)
    
        if debug:
            print(f"  Intersection: {intersection if 'intersection' in locals() else 0}")
            print(f"  Calculated Dice: {dc}")
    
        return dc

    def compute(self):
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()

class AbsoluteVolumeDifferenceNative(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        # Registered buffers are created on CPU by default.
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        # Disable gradient tracking for evaluation metrics.
        with torch.no_grad():
            # Immediately detach and move inputs to CPU.
            y_pred_cpu = y_pred.detach().cpu()
            y_true_cpu = y_true.detach().cpu()

            # Binarize predictions and ground truth.
            y_pred_binary = (y_pred_cpu > 0.5).float()
            y_true_binary = (y_true_cpu > 0.5).float()

            # Ensure voxel_size is on CPU.
            voxel_size_cpu = self.voxel_size.cpu()

            # Compute volumes: total voxel count multiplied by the voxel volume.
            ground_truth_volume = torch.sum(y_true_binary) * voxel_size_cpu
            prediction_volume = torch.sum(y_pred_binary) * voxel_size_cpu

            # Compute absolute volume difference.
            abs_vol_diff = torch.abs(ground_truth_volume - prediction_volume)

            # Update the running sum and count (all on CPU).
            self.sum += abs_vol_diff
            self.count += 1

            return abs_vol_diff

    def compute(self):
        # Compute the average absolute volume difference over all processed batches.
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach().cpu()
        else:
            return torch.tensor(0.0, device='cpu')

    def reset(self):
        # Reset the running totals.
        self.sum.zero_()
        self.count.zero_()

class AbsoluteLesionCountDifferenceCC3D(nn.Module):
    def __init__(self, connectivity=26, **kwargs):
        """
        Computes the absolute difference between the number of lesions (i.e. connected components)
        in the prediction and ground truth, averaged over the evaluated samples.
        
        Parameters:
            connectivity (int): Connectivity used for defining lesions (default 26 for 3D).
        """
        super().__init__()
        self.register_buffer('total_ald', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.connectivity = connectivity

    def forward(self, y_pred, y_true):
        """
        Parameters:
            y_pred (Tensor): Predicted segmentation tensor of shape [B, 1, H, W, D]
            y_true (Tensor): Ground truth segmentation tensor of shape [B, 1, H, W, D]
            
        Returns:
            Tensor: The average absolute lesion count difference for the batch.
        """
        with torch.no_grad():
            if y_pred.ndim == 5 and y_pred.shape[0] == 1:
                y_pred = y_pred.squeeze(0)
                y_true = y_true.squeeze(0)
            # Move tensors to CPU and detach.
            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            
            # Binarize the predictions and ground truth (threshold at 0.5)
            y_pred_np = (y_pred_np > 0.5).astype(np.uint8)
            y_true_np = (y_true_np > 0.5).astype(np.uint8)

            # After decollation, input is a single sample with shape [C, H, W, D].
            # We must select the 3D volume for cc3d.
            if y_pred_np.ndim != 4 or y_true_np.ndim != 4:
                raise ValueError(f"Expected 4D input [C,H,W,D], but got pred: {y_pred_np.shape}, true: {y_true_np.shape}")
            
            # Extract 3D volumes. Assuming single-channel output (C=1).
            true_vol = y_true_np[0, :, :, :]
            pred_vol = y_pred_np[0, :, :, :]

            # Count connected components (lesions) in the ground truth.
            _, num_true = cc3d.connected_components(true_vol, 
                                                     connectivity=self.connectivity, 
                                                     return_N=True)
            # Count connected components (lesions) in the prediction.
            _, num_pred = cc3d.connected_components(pred_vol, 
                                                     connectivity=self.connectivity, 
                                                     return_N=True)
            # Compute the absolute difference in lesion count.
            diff = abs(num_pred - num_true)
            
            # Update the running total and count for this single sample.
            self.total_ald += diff
            self.count += 1
            
            # Return the result as a torch tensor on the appropriate device.
            return torch.tensor(diff, device=self.total_ald.device)

    def compute(self):
        """
        Returns the accumulated average absolute lesion count difference.
        """
        if self.count > 0:
            result = self.total_ald / self.count.float()
            return result.detach()
        else:
            return torch.tensor(0.0, device=self.total_ald.device)

    def reset(self):
        """
        Resets the running totals.
        """
        self.total_ald.zero_()
        self.count.zero_()

class LesionF1CC3DScore(nn.Module):
    def __init__(self, epsilon=1e-6, connectivity=26, empty_value=1.0):
        super().__init__()
        # Registered buffers remain on CPU.
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon
        self.connectivity = connectivity
        self.empty_value = empty_value

    def forward(self, y_pred, y_true):
        # Disable gradient tracking for evaluation metrics.
        with torch.no_grad():
            if y_pred.ndim == 5 and y_pred.shape[0] == 1:
                y_pred = y_pred.squeeze(0)
                y_true = y_true.squeeze(0)
            # Immediately detach and move to CPU.
            y_pred_cpu = y_pred.detach().cpu()
            y_true_cpu = y_true.detach().cpu()
            
            # Binarize predictions and ground truth.
            y_pred_binary = (y_pred_cpu > 0.5).float()
            y_true_binary = (y_true_cpu > 0.5).float()
            
            # Convert tensors to NumPy arrays.
            y_pred_np = y_pred_binary.numpy()
            y_true_np = y_true_binary.numpy()
            
            # Clean up CPU tensors that are no longer needed.
            del y_pred_cpu, y_true_cpu, y_pred_binary, y_true_binary
            
            # After decollation, input is a single sample with shape [C, H, W, D].
            if y_pred_np.ndim != 4 or y_true_np.ndim != 4:
                raise ValueError(f"Expected 4D input [C,H,W,D], but got pred: {y_pred_np.shape}, true: {y_true_np.shape}")

            # Compute lesion F1 score for the single sample. Extract 3D volume.
            sample_f1 = self.compute_lesion_f1_score(y_true_np[0], y_pred_np[0])
            
            # Convert the result to a torch tensor on CPU.
            mean_f1_score_tensor = torch.tensor(sample_f1, device='cpu')
            
            # Update running totals.
            self.sum += mean_f1_score_tensor
            self.count += 1
            
            # Clean up the NumPy arrays if needed.
            del y_pred_np, y_true_np
            
            return mean_f1_score_tensor

    def compute_lesion_f1_score(self, ground_truth, prediction):
        """
        Computes the lesion F1 score for a single 3D volume using connected component analysis.
        
        Parameters:
            ground_truth (np.array): 3D binary mask for ground truth.
            prediction (np.array): 3D binary mask for prediction.
            
        Returns:
            f1_score (float): The computed lesion F1 score.
        """
        tp = 0
        fp = 0
        fn = 0

        # Compute the intersection between ground truth and prediction.
        intersection = np.logical_and(ground_truth, prediction)
        
        # Connected components on ground truth.
        labeled_ground_truth, num_gt = cc3d.connected_components(
            ground_truth, connectivity=self.connectivity, return_N=True
        )
        if num_gt > 0:
            for _, binary_cluster in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
                if np.logical_and(binary_cluster, intersection).any():
                    tp += 1
                else:
                    fn += 1

        # Connected components on prediction.
        labeled_prediction, num_pred = cc3d.connected_components(
            prediction, connectivity=self.connectivity, return_N=True
        )
        if num_pred > 0:
            for _, binary_cluster in cc3d.each(labeled_prediction, binary=True, in_place=True):
                if not np.logical_and(binary_cluster, ground_truth).any():
                    fp += 1

        # Special case: if no lesions are present in both.
        if tp + fp + fn == 0:
            _, num_gt_check = cc3d.connected_components(ground_truth, connectivity=self.connectivity, return_N=True)
            if num_gt_check == 0:
                return self.empty_value

        # Compute the lesion F1 score.
        f1_score = tp / (tp + (fp + fn) / 2 + self.epsilon)
        return f1_score

    def compute(self):
        # Return the running average of the lesion F1 score.
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach().cpu()
        else:
            return torch.tensor(0.0, device='cpu')

    def reset(self):
        # Reset the running totals.
        self.sum.zero_()
        self.count.zero_()


class HausdorffDistance95Medpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))

    def forward(self, y_pred, y_true):
        # Disable gradient tracking for evaluation metrics.
        with torch.no_grad():
            if y_pred.ndim == 5 and y_pred.shape[0] == 1:
                y_pred = y_pred.squeeze(0)
                y_true = y_true.squeeze(0)

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            
            if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
                # Both prediction and ground truth are empty, consider it perfect match
                hd95 = 0.0
            elif np.sum(y_true) > 0 and np.sum(y_pred) > 0:
                # Both have positive pixels, compute HD95
                hd95 = metric.binary.hd95(y_pred, y_true)
            else:
                # One is empty and the other is not, consider it worst case
                hd95 = np.sqrt(np.sum(np.array(y_true.shape)**2))  # diagonal of the volume
            
            self.sum += hd95
            self.count += 1
            return torch.tensor(hd95, device=self.sum.device)

    def compute(self):
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()

class HausdorffDistance95Native(nn.Module):
    def __init__(self):
        super().__init__()
        # Registered buffers remain on CPU.
        self.register_buffer('total_hd95', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        
    def forward(self, y_pred, y_true):
        # Disable gradient tracking for evaluation.
        with torch.no_grad():
            if y_pred.ndim == 5 and y_pred.shape[0] == 1:
                y_pred = y_pred.squeeze(0)
                y_true = y_true.squeeze(0)
            # Detach inputs and move to CPU immediately.
            y_pred_cpu = y_pred.detach().cpu()
            y_true_cpu = y_true.detach().cpu()
            
            # Convert tensors to NumPy arrays.
            y_pred_np = y_pred_cpu.numpy()
            y_true_np = y_true_cpu.numpy()
            
            # Clean up CPU tensors.
            del y_pred_cpu, y_true_cpu
            
            # Remove singleton channel dimension if present.
            if y_pred_np.ndim == 4 and y_pred_np.shape[-1] == 1:
                y_pred_np = np.squeeze(y_pred_np, axis=-1)
            if y_true_np.ndim == 4 and y_true_np.shape[-1] == 1:
                y_true_np = np.squeeze(y_true_np, axis=-1)
            
            # Binarize the predictions and ground truth.
            y_pred_np = (y_pred_np > 0.5).astype(np.uint8)
            y_true_np = (y_true_np > 0.5).astype(np.uint8)
            
            # If batched (shape [B, H, W, D]), compute HD95 for each sample.
            if y_pred_np.ndim == 4:
                hd95_list = []
                batch_size = y_pred_np.shape[0]
                for i in range(batch_size):
                    sample_hd95 = self.compute_hd95(y_true_np[i], y_pred_np[i])
                    hd95_list.append(sample_hd95)
                hd95 = float(np.mean(hd95_list))
                del hd95_list  # Free the list.
            elif y_pred_np.ndim == 3:
                hd95 = self.compute_hd95(y_true_np, y_pred_np)
            else:
                raise RuntimeError(f"[HD95] Unexpected input dimensions: {y_pred_np.ndim}")
            
            # Clean up NumPy arrays.
            del y_pred_np, y_true_np
            
            # Update running totals (buffers remain on CPU).
            self.total_hd95 += hd95
            self.count += 1
            
            # Return the HD95 result as a CPU tensor.
            return torch.tensor(hd95, device=self.total_hd95.device)
    
    def compute_hd95(self, ground_truth, prediction):
        """
        Computes the 95th percentile Hausdorff distance between two 3D binary masks.
        
        Parameters:
            ground_truth (np.array): 3D binary mask for ground truth.
            prediction (np.array): 3D binary mask for prediction.
            
        Returns:
            hd95 (float): The 95th percentile Hausdorff distance.
        """
        # Case when both masks are empty: perfect match.
        if np.sum(ground_truth) == 0 and np.sum(prediction) == 0:
            return 0.0
        # If one mask is empty and the other is not, return a worst-case distance (diagonal length).
        if np.sum(ground_truth) == 0 or np.sum(prediction) == 0:
            diag = np.sqrt(np.sum(np.array(ground_truth.shape) ** 2))
            return diag
        
        # Define a 3D connectivity structure (26-connectivity).
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        
        # Ensure both inputs are 3D.
        if ground_truth.ndim != 3 or prediction.ndim != 3:
            raise RuntimeError(f"[HD95] Input masks must be 3D. Got ground_truth.ndim={ground_truth.ndim}, prediction.ndim={prediction.ndim}")
        
        # Compute the boundary (surface) of the ground truth.
        try:
            gt_eroded = binary_erosion(ground_truth, structure=structure)
        except Exception as e:
            raise RuntimeError(f"[HD95] Error during binary_erosion for ground_truth with shape {ground_truth.shape}: {e}")
        gt_border = ground_truth - gt_eroded
        
        # Compute the boundary (surface) of the prediction.
        try:
            pred_eroded = binary_erosion(prediction, structure=structure)
        except Exception as e:
            raise RuntimeError(f"[HD95] Error during binary_erosion for prediction with shape {prediction.shape}: {e}")
        pred_border = prediction - pred_eroded
        
        # Compute the distance transforms on the complement (background) of each mask.
        dt_gt = distance_transform_edt(1 - ground_truth)
        dt_pred = distance_transform_edt(1 - prediction)
        
        # Get distances for each boundary voxel.
        distances_pred_to_gt = dt_gt[pred_border.astype(bool)]
        distances_gt_to_pred = dt_pred[gt_border.astype(bool)]
        
        # Combine the distances.
        all_distances = np.concatenate((distances_pred_to_gt, distances_gt_to_pred))
        if all_distances.size == 0:
            return 0.0
        
        hd95 = np.percentile(all_distances, 95)
        return float(hd95)
    
    def compute(self):
        if self.count > 0:
            result = self.total_hd95 / self.count.float()
            return result.detach().cpu()
        else:
            return torch.tensor(0.0, device='cpu')
    
    def reset(self):
        self.total_hd95.zero_()
        self.count.zero_()

class HausdorffDistance95MonaiMm(nn.Module):
    """
    Computes the 95th percentile Hausdorff Distance in millimeters (mm) using MONAI as a backend.
    This metric is designed for single-class segmentation (foreground vs. background).
    
    Parameters:
        spacing (tuple or list): The voxel spacing in (x, y, z) order.
    """
    def __init__(self, spacing, **kwargs):
        super().__init__()
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("`spacing` must be a tuple or list of 3 floats (x, y, z).")
        self.spacing = spacing
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))

    def forward(self, y_pred, y_true):
        """
        Computes the HD95 for a batch of predictions and ground truths.

        Parameters:
            y_pred (Tensor): Predicted segmentation tensor, shape [B, 1, H, W, D].
            y_true (Tensor): Ground truth segmentation tensor, shape [B, 1, H, W, D].
        """
        with torch.no_grad():
            y_pred = y_pred.to(y_true.device)
            
            y_pred_bin = (y_pred > 0.5).int()
            y_true_bin = y_true.int()

            batch_size = y_pred_bin.shape[0]
            hd_scores = []

            for i in range(batch_size):
                pred_i = y_pred_bin[i]
                true_i = y_true_bin[i]

                pred_is_empty = torch.sum(pred_i) == 0
                true_is_empty = torch.sum(true_i) == 0

                if pred_is_empty and true_is_empty:
                    hd_scores.append(0.0)
                    continue
                
                if pred_is_empty or true_is_empty:
                    # If one is empty, MONAI returns NaN. We return a penalty (image diagonal).
                    diag = np.sqrt(np.sum( (np.array(true_i.shape[1:]) * np.array(self.spacing))**2 ))
                    hd_scores.append(diag)
                    continue

                # Add batch dim back for one_hot conversion
                pred_i_batch = pred_i.unsqueeze(0)
                true_i_batch = true_i.unsqueeze(0)

                # Convert to one-hot format [B, C, H, W, D] where C=2 (bg, fg)
                pred_one_hot = one_hot(pred_i_batch, num_classes=2)
                true_one_hot = one_hot(true_i_batch, num_classes=2)

                hd_val = compute_hausdorff_distance(
                    y_pred=pred_one_hot,
                    y=true_one_hot,
                    include_background=False,
                    percentile=95,
                    spacing=self.spacing
                ).item()
                
                hd_scores.append(hd_val)
            
            batch_hd_mean = torch.tensor(np.mean(hd_scores), device=y_pred.device, dtype=torch.float)

            self.sum += batch_hd_mean * batch_size
            self.count += batch_size

            return batch_hd_mean

    def compute(self):
        """
        Returns the running average of the HD95 score.
        """
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        """
        Resets the running totals.
        """
        self.sum.zero_()
        self.count.zero_()

class SurfaceDiceMonai(nn.Module):
    """
    Computes the Normalized Surface Dice (NSD) using MONAI as a backend.
    This metric evaluates the agreement between segmentation boundaries.
    
    Parameters:
        spacing (tuple or list): The voxel spacing in (x, y, z) order.
        tolerance_mm (float): The tolerance in millimeters (mm) for considering
                              a surface voxel correctly segmented. Defaults to 1.0 mm.
    """
    def __init__(self, spacing, tolerance_mm=1.0, **kwargs):
        super().__init__()
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("`spacing` must be a tuple or list of 3 floats (x, y, z).")
        self.spacing = spacing
        self.tolerance_mm = tolerance_mm
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))

    def forward(self, y_pred, y_true):
        """
        Computes the Surface Dice for a batch of predictions and ground truths.

        Parameters:
            y_pred (Tensor): Predicted segmentation tensor, shape [B, 1, H, W, D].
            y_true (Tensor): Ground truth segmentation tensor, shape [B, 1, H, W, D].
        """
        with torch.no_grad():
            y_pred = y_pred.to(y_true.device)

            y_pred_bin = (y_pred > 0.5).int()
            y_true_bin = y_true.int()

            batch_size = y_pred_bin.shape[0]
            sd_scores = []

            for i in range(batch_size):
                pred_i = y_pred_bin[i]
                true_i = y_true_bin[i]

                pred_is_empty = torch.sum(pred_i) == 0
                true_is_empty = torch.sum(true_i) == 0

                if pred_is_empty and true_is_empty:
                    # Both are empty, which is a perfect match for surface dice.
                    sd_scores.append(1.0)
                    continue
                
                if pred_is_empty or true_is_empty:
                    # One is empty, a complete mismatch.
                    sd_scores.append(0.0)
                    continue
                
                # Add batch dim back for one_hot conversion
                pred_i_batch = pred_i.unsqueeze(0)
                true_i_batch = true_i.unsqueeze(0)

                # Convert to one-hot format [B, C, H, W, D] where C=2 (bg, fg)
                pred_one_hot = one_hot(pred_i_batch, num_classes=2)
                true_one_hot = one_hot(true_i_batch, num_classes=2)

                # MONAI's class_thresholds expects a list of thresholds, one per class.
                # Since we exclude the background, we only need one for our foreground class.
                sd_val = compute_surface_dice(
                    y_pred=pred_one_hot,
                    y=true_one_hot,
                    class_thresholds=[self.tolerance_mm],
                    include_background=False,
                    spacing=self.spacing
                ).item()

                sd_scores.append(sd_val)

            batch_sd_mean = torch.tensor(np.mean(sd_scores), device=y_pred.device, dtype=torch.float)

            self.sum += batch_sd_mean * batch_size
            self.count += batch_size

            return batch_sd_mean

    def compute(self):
        """
        Returns the running average of the Surface Dice score.
        """
        if self.count > 0:
            result = self.sum / self.count.float()
            return result.detach()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        """
        Resets the running totals.
        """
        self.sum.zero_()
        self.count.zero_()

class _VoxelConfusionBase(nn.Module):
    """Internal base class for voxel-level confusion matrix metrics."""
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))

    def _get_binary_masks(self, y_pred, y_true):
        y_pred = y_pred.to(y_true.device)
        y_pred_bin = (y_pred > 0.5).bool()
        y_true_bin = (y_true > 0.5).bool()
        return y_pred_bin, y_true_bin

    def forward(self, y_pred, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute(self):
        """Returns the average count per sample over all processed batches."""
        if self.count > 0:
            return self.sum / self.count.float()
        return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        """Resets the running totals."""
        self.sum.zero_()
        self.count.zero_()


class VoxelTruePositives(_VoxelConfusionBase):
    """Computes the total number of True Positive voxels."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred_bin, y_true_bin = self._get_binary_masks(y_pred, y_true)
            value = torch.sum(y_pred_bin & y_true_bin).float()
            self.sum += value
            self.count += 1
            return value


class VoxelFalsePositives(_VoxelConfusionBase):
    """Computes the total number of False Positive voxels."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred_bin, y_true_bin = self._get_binary_masks(y_pred, y_true)
            value = torch.sum(y_pred_bin & ~y_true_bin).float()
            self.sum += value
            self.count += 1
            return value


class VoxelFalseNegatives(_VoxelConfusionBase):
    """Computes the total number of False Negative voxels."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred_bin, y_true_bin = self._get_binary_masks(y_pred, y_true)
            value = torch.sum(~y_pred_bin & y_true_bin).float()
            self.sum += value
            self.count += 1
            return value


class VoxelTrueNegatives(_VoxelConfusionBase):
    """Computes the total number of True Negative voxels."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred_bin, y_true_bin = self._get_binary_masks(y_pred, y_true)
            value = torch.sum(~y_pred_bin & ~y_true_bin).float()
            self.sum += value
            self.count += 1
            return value


class _VolumeMm3Base(nn.Module):
    """Internal base class for volume calculation metrics."""
    def __init__(self, spacing, **kwargs):
        super().__init__()
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("`spacing` must be a tuple or list of 3 floats (x, y, z).")
        
        self.voxel_volume = np.prod(spacing)
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))

    def forward(self, y_pred, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute(self):
        """Returns the average volume in mm³ over all processed batches."""
        if self.count > 0:
            return self.sum / self.count.float()
        return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        """Resets the running totals."""
        self.sum.zero_()
        self.count.zero_()


class PredictedVolumeMm3(_VolumeMm3Base):
    """Computes the volume of the predicted segmentation in cubic millimeters (mm³)."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            # This class only uses y_pred for its calculation.
            y_pred_bin = (y_pred > 0.5)
            
            voxel_count = torch.sum(y_pred_bin).float()
            volume = voxel_count * self.voxel_volume
            
            self.sum += volume
            self.count += 1
            return volume


class GroundTruthVolumeMm3(_VolumeMm3Base):
    """Computes the volume of the ground truth segmentation in cubic millimeters (mm³)."""
    def forward(self, y_pred, y_true):
        with torch.no_grad():
            # This class only uses y_true for its calculation.
            y_true_bin = (y_true > 0.5)

            voxel_count = torch.sum(y_true_bin).float()
            volume = voxel_count * self.voxel_volume

            self.sum += volume
            self.count += 1
            return volume


# ==================== 2D SLICE-BASED METRICS ====================

class Dice2DForegroundOnly(nn.Module):
    """
    2D Dice coefficient that only computes metrics for slices with foreground pixels.
    Ignores empty slices to avoid bias from class imbalance.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()
            
            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            # Only compute dice for slices with foreground pixels
            has_foreground = torch.sum(y_true_binary) > 0
            
            if has_foreground:
                intersection = torch.sum(y_pred_binary * y_true_binary)
                cardinality_pred = torch.sum(y_pred_binary)
                cardinality_true = torch.sum(y_true_binary)
                
                dice = (2.0 * intersection + self.epsilon) / (cardinality_pred + cardinality_true + self.epsilon)
                
                self.sum += dice
                self.count += 1
                
                return dice
            else:
                # Return 0 for empty slices but don't count them
                return torch.tensor(0.0, device='cpu')

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device='cpu')

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class VoxelPrecision2D(nn.Module):
    """
    Computes 2D slice-wise precision (positive predictive value).
    Precision = TP / (TP + FP)
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            tp = torch.sum(y_pred_binary * y_true_binary).float()
            fp = torch.sum(y_pred_binary * (1 - y_true_binary)).float()
            
            precision = tp / (tp + fp + self.epsilon)
            
            self.sum += precision
            self.count += 1
            
            return precision

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class VoxelSensitivity2D(nn.Module):
    """
    Computes 2D slice-wise sensitivity (recall/true positive rate).
    Sensitivity = TP / (TP + FN)
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            tp = torch.sum(y_pred_binary * y_true_binary).float()
            fn = torch.sum((1 - y_pred_binary) * y_true_binary).float()
            
            sensitivity = tp / (tp + fn + self.epsilon)
            
            self.sum += sensitivity
            self.count += 1
            
            return sensitivity

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class VoxelSpecificity2D(nn.Module):
    """
    Computes 2D slice-wise specificity (true negative rate).
    Specificity = TN / (TN + FP)
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            tn = torch.sum((1 - y_pred_binary) * (1 - y_true_binary)).float()
            fp = torch.sum(y_pred_binary * (1 - y_true_binary)).float()
            
            specificity = tn / (tn + fp + self.epsilon)
            
            self.sum += specificity
            self.count += 1
            
            return specificity

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class VoxelF1Score2D(nn.Module):
    """
    Computes 2D slice-wise F1 score.
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            tp = torch.sum(y_pred_binary * y_true_binary).float()
            fp = torch.sum(y_pred_binary * (1 - y_true_binary)).float()
            fn = torch.sum((1 - y_pred_binary) * y_true_binary).float()
            
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            
            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
            
            self.sum += f1
            self.count += 1
            
            return f1

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class VoxelF2Score2D(nn.Module):
    """
    Computes 2D slice-wise F2 score (emphasizes recall over precision).
    F2 = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    where beta = 2 (emphasizes recall 4x more than precision)
    """
    def __init__(self, beta=2.0, epsilon=1e-6):
        super().__init__()
        self.register_buffer('sum', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.beta = beta
        self.beta_squared = beta * beta
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            y_pred_binary = (y_pred > 0.5).float()
            y_true_binary = (y_true > 0.5).float()

            tp = torch.sum(y_pred_binary * y_true_binary).float()
            fp = torch.sum(y_pred_binary * (1 - y_true_binary)).float()
            fn = torch.sum((1 - y_pred_binary) * y_true_binary).float()
            
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            
            f2 = (1 + self.beta_squared) * (precision * recall) / (self.beta_squared * precision + recall + self.epsilon)
            
            self.sum += f2
            self.count += 1
            
            return f2

    def compute(self):
        if self.count > 0:
            return self.sum / self.count.float()
        else:
            return torch.tensor(0.0, device=self.sum.device)

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class SliceWiseMetricsAggregator(nn.Module):
    """
    Aggregates multiple slice-wise metrics and provides comprehensive statistics.
    Tracks both per-slice statistics and overall averages.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('total_slices', torch.tensor(0))
        self.register_buffer('foreground_slices', torch.tensor(0))
        self.register_buffer('empty_slices', torch.tensor(0))
        
        # Individual metrics
        self.dice_fg = Dice2DForegroundOnly()
        self.precision = VoxelPrecision2D()
        self.sensitivity = VoxelSensitivity2D()
        self.specificity = VoxelSpecificity2D()
        self.f1_score = VoxelF1Score2D()
        self.f2_score = VoxelF2Score2D()

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            self.total_slices += 1
            
            # Check if slice has foreground
            has_foreground = torch.sum(y_true > 0.5) > 0
            if has_foreground:
                self.foreground_slices += 1
            else:
                self.empty_slices += 1
            
            # Compute all metrics
            results = {
                'dice_2d_fg': self.dice_fg(y_pred, y_true),
                'precision_2d': self.precision(y_pred, y_true),
                'sensitivity_2d': self.sensitivity(y_pred, y_true),
                'specificity_2d': self.specificity(y_pred, y_true),
                'f1_2d': self.f1_score(y_pred, y_true),
                'f2_2d': self.f2_score(y_pred, y_true),
            }
            
            return results

    def compute(self):
        results = {
            'dice_2d_fg': self.dice_fg.compute(),
            'precision_2d': self.precision.compute(),
            'sensitivity_2d': self.sensitivity.compute(),
            'specificity_2d': self.specificity.compute(),
            'f1_2d': self.f1_score.compute(),
            'f2_2d': self.f2_score.compute(),
            'total_slices': self.total_slices.float(),
            'foreground_slices': self.foreground_slices.float(),
            'empty_slices': self.empty_slices.float(),
            'foreground_ratio': self.foreground_slices.float() / (self.total_slices.float() + 1e-6)
        }
        return results

    def reset(self):
        self.total_slices.zero_()
        self.foreground_slices.zero_()
        self.empty_slices.zero_()
        
        self.dice_fg.reset()
        self.precision.reset()
        self.sensitivity.reset()
        self.specificity.reset()
        self.f1_score.reset()
        self.f2_score.reset()

def get_metric(name, params):
    if name not in globals():
        raise ValueError(f"Metric class '{name}' not found in metrics.py")
    metric_class = globals()[name]
    return metric_class(**params)
