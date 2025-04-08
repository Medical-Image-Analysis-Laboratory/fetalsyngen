#from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.neighbors import NearestNeighbors  # Intel optimized NearestNeighbors
import torch
import torch.nn.functional as F
from skimage.morphology import ball
from scipy.ndimage import label, binary_dilation, binary_erosion
import random
from fetalsyngen.generator.alterations.brainAlterations import brain_Alterations
import nibabel as nib
import numpy as np




class cc_Alterations:        
    def __init__(self, alteration_prob: float, target_label: int,
                 max_dilation: int, min_dilation: int,
                 min_erosion: int, max_erosion: int,
                 min_posterior_loss: float, max_posterior_loss: float,
                 min_anterior_loss: float, max_anterior_loss: float,
                 min_amplitude_kinked: int, max_amplitude_kinked: int,
                 min_freq_kinked: int, max_freq_kinked: int, 
                 cc_alter_prob: torch.Tensor,
                 brain_alterations: brain_Alterations | None = None):
        """
        Initialize the transformation class with parameters for different types of alterations.

        Args:
            alteration_prob (float): Probability of applying an alteration to the segmentation.
            target_label (int): Label of the target region to be modified.
            max_dilation (int): Maximum dilation size for region expansion.
            min_dilation (int): Minimum dilation size for region expansion.
            min_erosion (int): Minimum erosion size for shrinking the region.
            max_erosion (int): Maximum erosion size for shrinking the region.
            min_posterior_loss (float): Minimum percentage of posterior region loss.
            max_posterior_loss (float): Maximum percentage of posterior region loss.
            min_anterior_loss (float): Minimum percentage of anterior region loss.
            max_anterior_loss (float): Maximum percentage of anterior region loss.
            min_amplitude_kinked (int): Minimum amplitude for kinked deformation.
            max_amplitude_kinked (int): Maximum amplitude for kinked deformation.
            min_freq_kinked (int): Minimum frequency for kinked deformation.
            max_freq_kinked (int): Maximum frequency for kinked deformation.
        """
        self.alteration_prob = alteration_prob
        self.target_label = target_label
        self.max_dilation = max_dilation
        self.min_dilation = min_dilation
        self.min_erosion = min_erosion
        self.max_erosion = max_erosion
        self.min_posterior_loss = min_posterior_loss
        self.max_posterior_loss = max_posterior_loss
        self.min_anterior_loss = min_anterior_loss
        self.max_anterior_loss = max_anterior_loss
        self.min_amplitude_kinked = min_amplitude_kinked
        self.max_amplitude_kinked = max_amplitude_kinked
        self.min_freq_kinked = min_freq_kinked
        self.max_freq_kinked = max_freq_kinked
        self.brain_alteration = brain_alterations
        self.cc_alter_prob = torch.tensor(cc_alter_prob, dtype=torch.float32)



    def numpy_to_tensor(self, array):
        """Convert a NumPy array to a PyTorch tensor."""
        return torch.tensor(array, dtype=torch.int64)


    def tensor_to_numpy(self, tensor):
        """Convert a PyTorch tensor to a NumPy array."""
        return tensor.cpu().numpy()
    
    def brain_volumne_computation(self, segm, segm_affine):
        """Computes brain volume in dm3."""
        voxel_size = torch.abs(torch.linalg.det(segm_affine[:3, :3]))
        brain_tissues = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        brain_mask = torch.isin(segm, brain_tissues)

        brain_voxels = brain_mask.sum()
        brain_volume_mm3 = brain_voxels * voxel_size
        brain_volume_dm3 = float(brain_volume_mm3 / 1000000)

        brain_indices = torch.nonzero(brain_mask, as_tuple=True)
        y_coords = brain_indices[1]
        brain_y_length = int(y_coords.max() - y_coords.min())

        return brain_volume_dm3, brain_y_length
    
    
    def max_ratio(self, segm, segm_affine):
        """Computes a scaling ratio based on a reference 38-week gestational age (GW) brain volume."""
        brain_volume, _ = self.brain_volumne_computation(segm, segm_affine)
        max_ratio = brain_volume/0.444 # Based on 38gw brain volume

        return max_ratio
    
    
    def length_ratio(self, segm, segm_affine):
        """Computes a scaling ratio based on a reference 21-week gestational age (GW) brain length."""
        _, brain_y_length = self.brain_volumne_computation(segm, segm_affine)
        length_ratio = 80/brain_y_length # Based on 21gw brain length

        return length_ratio


    def NN_interpolation(self, tensor, mask, dilated_mask=None, background=False):
        """Perform nearest neighbor interpolation on a given tensor or seed region.

        This function interpolates missing values using the nearest neighbor approach.
        It can be applied either to an entire segmentation tensor or to a specific seed 
        region within a dilated mask.

        Args:
            tensor (torch.Tensor): Input tensor representing either the segmentation or a binary mask.
            mask (torch.Tensor): Binary mask indicating missing values to fill.
            dilated_mask (torch.Tensor, optional): A dilated version of the mask, used for seed interpolation. Defaults to None.
            background (bool, optional): Whether to interpolate using background pixels. Defaults to False.

        Returns:
            - tensor (torch.Tensor): Interpolated tensor where missing values are filled.
        """
        if dilated_mask is not None:
            # Seed Interpolation Mode: Find missing values only in the dilated mask
            NN_tensor = mask.clone()
            missing_values = torch.nonzero((dilated_mask == 1) & (mask == 0))
            valid_indices = torch.nonzero(mask)
        else:
            # General Interpolation Mode: Work on the entire tensor
            NN_tensor = tensor.clone()
            NN_tensor[mask > 0.5] = 0  # Remove old labels
            missing_values = torch.nonzero(mask)
            valid_indices = torch.nonzero(mask == 0 if background else NN_tensor != 0)

        if valid_indices.numel() == 0 or missing_values.numel() == 0:
            return tensor  # No interpolation needed if no valid points exist

        # Convert to NumPy for faster processing
        missing_values_np = missing_values.cpu().numpy()
        valid_indices_np = valid_indices.cpu().numpy()

        # Fit Nearest Neighbors model
        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(valid_indices_np)
        nearest_indices = knn.kneighbors(missing_values_np, return_distance=False).flatten()

        # Assign values in a vectorized manner
        NN_tensor[missing_values[:, 0], missing_values[:, 1], missing_values[:, 2]] = \
            NN_tensor[valid_indices[nearest_indices, 0], valid_indices[nearest_indices, 1], valid_indices[nearest_indices, 2]]
        
        return NN_tensor


    def biggest_connected_component(self, target_mask, tensor):
        """Extract the biggest connected component from a binary mask and apply nearest neighbor interpolation.

        Args:
            target_mask (torch.Tensor): Binary mask representing the region of interest.
            tensor (torch.Tensor): Segmentation tensor to be modified.

        Returns:
            - component_mask (torch.Tensor): Mask of the largest connected component.
            - tensor (torch.Tensor): Updated segmentation tensor with the largest component preserved.
        """
        labeled_mask, _ = label(self.tensor_to_numpy(target_mask))
        labeled_mask = self.numpy_to_tensor(labeled_mask).to(torch.long)

        component_sizes = torch.bincount(labeled_mask.view(-1))
        # If only background, no CC, skip the cc alterations
        #if len(component_sizes) < 2: return target_mask, tensor, True
        sorted_indices = torch.argsort(component_sizes[1:], descending=True) + 1
        target_label = sorted_indices[0]
        component_mask = labeled_mask == target_label

        tensor = self.NN_interpolation(tensor, target_mask)
        tensor[component_mask] = self.target_label

        return component_mask, tensor, False
    

    def random_mask(self, target_mask, segmentation, NN_tensor, NN_seed, seed, genparams):
        """Generate a random mask transformation based on three options: original mask, hyperplasia or hypoplasia.

        Args:
            target_mask (torch.Tensor): Binary mask indicating the affected area.
            segmentation (torch.Tensor): Segmentation tensor.
            NN_tensor (torch.Tensor): Interpolated segmentation tensor.
            NN_seed (torch.Tensor): Seed tensor after interpolation.
            seed (torch.Tensor): Original seed tensor.
            genparams (dict): Dictionary containing transformation parameters.

        Returns:
            - selected_target_mask (torch.Tensor): Transformed target mask after applying random pathology.
        """
        genparams_copy = genparams.copy()
        eroded_target_mask, _, _, _ = self.hypoplasia(target_mask, NN_tensor, NN_seed, seed, genparams_copy)
        dilated_target_mask, _, _, _ = self.hyperplasia(target_mask, segmentation, seed, genparams_copy)
        selected_target_mask = torch.stack([eroded_target_mask, dilated_target_mask, target_mask])

        probabilities = torch.tensor([0.7, 0.1, 0.2])
        selected_index = torch.multinomial(probabilities, 1).item()

        return selected_target_mask[selected_index]


    def random_alteration(self, seed, segmentation, genparams: dict = {}):
        """
        Simulates specific alterations on a segmentation mask based on a probability threshold.

        This function applies different types of deformations (hyperplasia, hypoplasia, partial loss, kinked, or agenesis)
        to a given label of a segmentation mask. If specific generation parameters (`genparams`) are provided, it selects the 
        appropriate transformation. Otherwise, it randomly chooses a transformation based on predefined probabilities.

        Args:
            seed (torch.Tensor): The seed mask of the target structure,, representing the initial state before transformation.
            segmentation (torch.Tensor): The segmentation mask of the target structure, representing the initial state before transformation.
            genparams (dict, optional): Dictionary containing parameters for specific transformations. Defaults to {}.

        Returns:
            - None: Placeholder return value.
            - transformed_segmentation (torch.Tensor): The altered segmentation after applying the pathology.
            - transformed_seed (torch.Tensor): The altered seed after applying the pathology.
            - genparams (dict): Updated generation parameters after transformation.
        """
        if self.alteration_prob > 0.5:
            target_mask = segmentation == self.target_label
            target_mask, segmentation, indicator = self.biggest_connected_component(target_mask, segmentation)
            
            if indicator:
                if self.brain_alteration is not None:
                    transformed_segmentation, transformed_seed, genparams = self.brain_alteration.random_brain_alteration(
                        seed=seed,
                        segmentation=segmentation,
                        genparams=genparams
                    )
                    return transformed_segmentation, transformed_seed, genparams
                else:
                    return segmentation, seed, genparams
            
            else:
                NN_segm = self.NN_interpolation(segmentation, target_mask)
                NN_seed = self.NN_interpolation(seed, target_mask)
                selected_target_mask = self.random_mask(target_mask, segmentation, NN_segm, NN_seed, seed, genparams)

                if 'kinked_amplitude' in genparams or 'kinked_freq' in genparams:
                    _, transformed_segmentation, transformed_seed, genparams = self.kinked(selected_target_mask, segmentation, seed, genparams)
                elif 'size_dilation' in genparams:
                    _, transformed_segmentation, transformed_seed, genparams = self.hyperplasia(selected_target_mask, segmentation, seed, genparams)
                elif 'size_erosion' in genparams:
                    _, transformed_segmentation, transformed_seed, genparams = self.hypoplasia(selected_target_mask, NN_segm, NN_seed, seed, genparams)
                elif 'anterior_loss' in genparams or 'posterior_loss' in genparams:
                    _, transformed_segmentation, transformed_seed, genparams = self.partial_loss(selected_target_mask, NN_segm, NN_seed, seed, genparams)
                elif 'angenesis' in genparams:
                    _, transformed_segmentation, transformed_seed, genparams = self.partial_loss(selected_target_mask, NN_segm, NN_seed, seed, genparams)
                else:
                    pathology_functions = [
                        self.hyperplasia,
                        self.hypoplasia,
                        self.partial_loss,
                        self.kinked,
                        self.agenesis
                    ]
                    selected_index = torch.multinomial(self.cc_alter_prob, 1).item()
                    selected_transformation = pathology_functions[selected_index]
                    
                    if selected_transformation == self.hyperplasia:
                        _, transformed_segmentation, transformed_seed, genparams = selected_transformation(target_mask, segmentation, seed, genparams)
                    elif selected_transformation == self.hypoplasia:
                        _, transformed_segmentation, transformed_seed, genparams = selected_transformation(target_mask, NN_segm, NN_seed, seed, genparams)
                    else:
                        selected_target_mask = self.random_mask(target_mask, segmentation, NN_segm, NN_seed, seed, genparams)
                        if selected_transformation == self.kinked:
                            _, transformed_segmentation, transformed_seed, genparams = selected_transformation(target_mask, segmentation, seed, genparams)
                        else:
                            _, transformed_segmentation, transformed_seed, genparams = selected_transformation(target_mask, NN_segm, NN_seed, seed, genparams)
                if self.brain_alteration is not None:
                    transformed_segmentation, transformed_seed, genparams = self.brain_alteration.random_brain_alteration(
                        seed=transformed_seed,
                        segmentation=transformed_segmentation,
                        genparams=genparams
                    )
                return transformed_segmentation, transformed_seed, genparams
        else:
            if self.brain_alteration is not None:
                transformed_segmentation, transformed_seed, genparams = self.brain_alteration.random_brain_alteration(
                    seed=seed,
                    segmentation=segmentation,
                    genparams=genparams
                )
                return transformed_segmentation, transformed_seed, genparams
            return segmentation, seed, genparams


    def hyperplasia(self, target_mask, segm, seed, genparams):
        """Apply hyperplasia transformation by dilating the affected region.

        Args:
            target_mask (torch.Tensor): Binary mask indicating the affected area.
            segm (torch.Tensor): Segmentation tensor.
            seed (torch.Tensor): Seed tensor.
            genparams (dict): Dictionary containing transformation parameters.

        Returns:
            - dilated_mask (torch.Tensor): Updated mask after dilation.
            - dilated_tensor (torch.Tensor): Updated segmentation tensor after dilation.
            - dilated_seed (torch.Tensor): Updated seed tensor after dilation.
            - genparams (dict): Updated transformation parameters.
        """
        # Struct of size 0 returns error
        self.max_dilation = round(self.max_dilation * self.max_ratio(segm, segm.affine))
        if (self.min_dilation == 0): self.min_dilation = 1
        if (self.max_dilation == 0): self.max_dilation = 1
        if self.min_dilation == self.max_dilation: size = self.min_dilation
        else:
            size = genparams.get('size_dilation', random.randint(self.min_dilation, self.max_dilation))
        genparams['size_dilation'] = size
        dilated_tensor = segm.clone()
        dilated_seed = seed.clone()

        NN_seed = torch.zeros_like(seed, dtype=torch.int64)
        NN_seed[target_mask] = seed[target_mask]

        struct = torch.tensor(ball(size))
        dilated_mask = torch.tensor(binary_dilation(self.tensor_to_numpy(target_mask), structure=self.tensor_to_numpy(struct)))

        extended_mask = self.NN_interpolation(seed, NN_seed, dilated_mask) 

        dilated_seed[dilated_mask] = extended_mask[dilated_mask]
        dilated_tensor[dilated_mask] = self.target_label

        return dilated_mask, dilated_tensor, dilated_seed, genparams


    def hypoplasia(self, target_mask, NN_segm, NN_seed, seed, genparams):
        """Apply hypoplasia transformation by eroding the affected region.

        Args:
            target_mask (torch.Tensor): Binary mask of the region to erode.
            NN_segm (torch.Tensor): Nearest-neighbor interpolated segmentation.
            NN_seed (torch.Tensor): Nearest-neighbor interpolated seed region.
            seed (torch.Tensor): Original seed segmentation.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - eroded_mask (torch.Tensor): Updated mask after erosion.
            - eroded_tensor (torch.Tensor): Updated segmentation tensor after erosion.
            - eroded_seed (torch.Tensor):  Updated seed segmentation tensor after erosion.
            - genparams (dict): Updated generation parameters.
        """
        size = genparams.get('size_erosion', random.randint(self.min_erosion, self.max_erosion))
        genparams['size_erosion'] = size

        struct = torch.ones((size, 1), dtype=torch.int).unsqueeze(0)
        eroded_mask = torch.tensor(binary_erosion(self.tensor_to_numpy(target_mask), structure=self.tensor_to_numpy(struct)))

        eroded_mask, NN_segm, _ = self.biggest_connected_component(eroded_mask, NN_segm)

        eroded_seed = NN_seed.clone()
        eroded_seed[eroded_mask] = seed[eroded_mask]
        eroded_tensor = NN_segm.clone()
        eroded_tensor[eroded_mask] = self.target_label

        return eroded_mask, eroded_tensor, eroded_seed, genparams
    

    def posterior_loss(self, min_y, total_length, coords, genparams):
        """Apply posterior loss by removing a random percentage of the posterior region.

        Args:
            min_y (int): Minimum y-coordinate of the affected region.
            total_length (int): Total extent of the affected region.
            coords (torch.Tensor): Coordinates of the affected region.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - genparams (dict): Updated generation parameters.
            - mask_condition (torch.Tensor): Boolean mask indicating the affected area.
        """
        percentage = genparams.get('posterior_loss', random.uniform(self.min_posterior_loss, self.max_posterior_loss))
        genparams['posterior_loss'] = percentage

        segment = min_y + int(percentage * total_length)
        mask_condition = (coords[:, 1] >= min_y) & (coords[:, 1] > segment)

        return genparams, mask_condition
    

    def anterior_loss(self, min_y, total_length, coords, genparams):
        """Apply anterior loss by removing a random percentage of the anterior region.

        Args:
            min_y (int): Minimum y-coordinate of the affected region.
            total_length (int): Total extent of the affected region.
            coords (torch.Tensor): Coordinates of the affected region.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - genparams (dict): Updated generation parameters.
            - mask_condition (torch.Tensor): Boolean mask indicating the affected area.
        """
        percentage = genparams.get('anterior_loss', random.uniform(self.min_anterior_loss, self.max_anterior_loss))
        genparams['anterior_loss'] = percentage

        segment = min_y + int(percentage * total_length)
        mask_condition = (coords[:, 1] >= min_y) & (coords[:, 1] < segment)

        return genparams, mask_condition


    def partial_loss(self, target_mask, NN_segm, NN_seed, seed, genparams):
        """Apply partial loss transformation by removing a part of the region.
        The function selects either anterior or posterior loss based on `genparams` or at random.

        Args:
            target_mask (torch.Tensor): Binary mask of the region to modify.
            NN_segm (torch.Tensor): Nearest-neighbor interpolated segmentation.
            NN_seed (torch.Tensor): Nearest-neighbor interpolated seed region.
            seed (torch.Tensor): Original seed segmentation.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - partial_loss_mask (torch.Tensor): Updated mask indicating the removed region.
            - partial_loss_segm (torch.Tensor): Updated segmentation after applying the loss.
            - partial_loss_seed (torch.Tensor): Updated seed segmentation after applying the loss.
            - genparams (dict): Updated generation parameters.
        """
        coords = torch.nonzero(target_mask, as_tuple=False)
        partial_loss_mask = torch.zeros_like(target_mask, dtype=torch.bool)

        if coords.size(0) > 0:
            min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
            total_length = max_y - min_y

            if 'anterior_loss' in genparams:
                selected_loss = self.anterior_loss
            elif 'posterior_loss' in genparams:
                selected_loss = self.posterior_loss
            else:
                selected_loss = random.choice([self.anterior_loss, self.posterior_loss])

            genparams, mask_condition = selected_loss(min_y, total_length, coords, genparams)
            segment_coords = coords[mask_condition]

            partial_loss_mask[segment_coords[:, 0], segment_coords[:, 1], segment_coords[:, 2]] = 1

        partial_loss_seed = NN_seed.clone()
        partial_loss_seed[partial_loss_mask] = seed[partial_loss_mask]
        partial_loss_segm = NN_segm.clone()
        partial_loss_segm[partial_loss_mask] = self.target_label

        return partial_loss_mask, partial_loss_segm, partial_loss_seed, genparams


    def kinked(self, target_mask, segm, seed, genparams):
        """Apply a sinusoidal deformation to the affected region to warp the x-coordinates, creating a 
        kinked effect.

        Args:
            target_mask (torch.Tensor): Binary mask of the region to modify.
            segm (torch.Tensor): Segmentation tensor.
            seed (torch.Tensor): Seed segmentation tensor. 
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - None: Placeholder return value.
            - kinked_segm (torch.Tensor): Updated segmentation after the kinked transformation.
            - kinked_seed (torch.Tensor): Updated seed segmentation after transformation.
            - genparams (dict): Updated generation parameters.
        """
        self.max_freq_kinked = round(self.max_freq_kinked * self.length_ratio(segm, segm.affine))
        roi_coords = torch.nonzero(target_mask, as_tuple=False)
        margin = 3 # Add a margin to include all the possible amplitude of the sinusoidal curve
        if roi_coords.size(0) > 0:
            min_z, min_y, min_x = roi_coords.min(dim=0).values
            max_z, max_y, max_x = roi_coords.max(dim=0).values

            roi_z, roi_y, roi_x = torch.meshgrid(
                torch.arange(min_z, max_z+margin, device=segm.device),
                torch.arange(min_y, max_y+margin, device=segm.device),
                torch.arange(min_x, max_x+margin, device=segm.device),
                indexing='ij'
            )

            amplitude = genparams['kinked_amplitude'] if 'kinked_amplitude' in genparams else random.randint(self.min_amplitude_kinked, self.max_amplitude_kinked)
            if self.min_freq_kinked > self.max_freq_kinked:
                self.max_freq_kinked = self.min_freq_kinked
            freq = genparams['kinked_freq'] if 'kinked_freq' in genparams else random.randint(self.min_freq_kinked, self.max_freq_kinked)
            frequency = freq * torch.pi / segm.shape[2]
            genparams['kinked_amplitude'] = amplitude
            genparams['kinked_freq'] = freq

            roi_x_warped = roi_x + amplitude * torch.sin(frequency * roi_y)

            # Normalize coordinates for grid_sample (-1 to 1 range)
            grid_z = (roi_z.float() / (segm.shape[0] - 1)) * 2 - 1
            grid_y = (roi_y.float() / (segm.shape[1] - 1)) * 2 - 1
            grid_x = (roi_x_warped.float() / (segm.shape[2] - 1)) * 2 - 1

            grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)
            seed_3d = seed.unsqueeze(0).unsqueeze(0).float()
            kinked_3d_roi_seed = F.grid_sample(seed_3d, grid, align_corners=True, mode='nearest')
            segm_3d = segm.unsqueeze(0).unsqueeze(0).float()
            kinked_3d_roi_segm = F.grid_sample(segm_3d, grid, align_corners=True, mode='nearest')
            
            kinked_seed = seed.clone()
            kinked_seed[min_z:max_z+margin, min_y:max_y+margin, min_x:max_x+margin] = kinked_3d_roi_seed.squeeze().round().long()
            kinked_segm = segm.clone()
            kinked_segm[min_z:max_z+margin, min_y:max_y+margin, min_x:max_x+margin] = kinked_3d_roi_segm.squeeze().round().long()

        return None, kinked_segm, kinked_seed, genparams


    def agenesis(self, target_mask, NN_segm, NN_seed, seed, genparams):
        """Simulate agenesis by completely removing the affected region.
        This function simply marks the agenesis condition in `genparams` but does not modify the 
        segmentation or seed, as the removal has already been handled in prior NN interpolation step.

        Args:
            target_mask (torch.Tensor): Binary mask of the region (not used in this function), placeholder argument.
            NN_segm (torch.Tensor): Nearest-neighbor interpolated segmentation.
            NN_seed (torch.Tensor): Nearest-neighbor interpolated seed region.
            seed (torch.Tensor): Original seed segmentation (not used in this function), placeholder argument.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            - None: Placeholder return value.
            - NN_segm (torch.Tensor): The nearest-neighbor interpolated segmentation remains unchanged.
            - NN_seed (torch.Tensor): The nearest-neighbor interpolated seed segmentation remains unchanged.
            - genparams (dict): Updated generation parameters with agenesis set to 1, as an boolean indicator that 
            agenesis transformation has been applied.
        """
        genparams['agenesis'] = 1
        return None, NN_segm, NN_seed, genparams