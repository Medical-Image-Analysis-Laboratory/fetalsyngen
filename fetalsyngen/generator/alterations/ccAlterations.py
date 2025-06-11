#from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.neighbors import NearestNeighbors  # Intel optimized NearestNeighbors
import torch
import torch.nn.functional as F
from skimage.morphology import ball
from scipy.ndimage import label, binary_dilation, binary_erosion
import random
from fetalsyngen.generator.alterations.brainAlterations import brain_Alterations
import numpy as np
import random


class cc_Alterations:        
    def __init__(self, alteration_prob: float, cc_threshold: float,
                 target_label: int, csf_label: int,
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
            cc_threshold (float): Threshold for percentage of pathological cases.
            target_label (int): Label of the target region to be modified.
            csf_label (int): Label for cerebrospinal fluid (CSF).
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
            cc_alter_prob (torch.Tensor): Class-specific probabilities for corpus callosum different alterations types.
            brain_alterations (brain_Alterations | None): Optional predefined brain alteration configurations.
        """
        self.alteration_prob = alteration_prob
        self.cc_threshold = cc_threshold
        self.target_label = target_label
        self.csf_label = csf_label
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
        """
        Computes the total brain volume and the anterior-posterior length 
        (in the Y-axis) from a segmentation map.

        Args:
            segm (torch.Tensor): 3D tensor representing the segmentation map.
            segm_affine (torch.Tensor): 4x4 affine matrix associated with the 
                segmentation image, used to calculate voxel dimensions.

        Returns:
            Tuple[float, int]:
                - brain_volume_dm3 (float): Computed brain volume in cubic decimeters (dm³).
                - brain_y_length (int): Length of the brain along the Y-axis in voxel units.
        """
        voxel_size = torch.abs(torch.linalg.det(segm_affine[:3, :3]))
        # Automatically get all non-background labels
        unique_labels = torch.unique(segm).int()
        brain_tissues = unique_labels[unique_labels != 0]
        brain_mask = torch.isin(segm, brain_tissues)

        brain_voxels = brain_mask.sum()
        brain_volume_mm3 = brain_voxels * voxel_size
        brain_volume_dm3 = float(brain_volume_mm3 / 1000000)

        brain_indices = torch.nonzero(brain_mask, as_tuple=True)
        y_coords = brain_indices[1]
        brain_y_length = int(y_coords.max() - y_coords.min())

        return brain_volume_dm3, brain_y_length
    
    
    def max_ratio(self, segm, segm_affine):
        """
        Computes the scaling ratio between the brain volume of the given segmentation 
        and a reference brain volume corresponding to a 38-week gestational age.

        Args:
            segm (torch.Tensor): 3D tensor of the brain segmentation.
            segm_affine (torch.Tensor): 4x4 affine matrix used to compute the voxel size.

        Returns:
            max_ratio (float): The ratio of the computed brain volume (in dm³) to the reference 
                volume of 0.444 dm³ (typical for a 38-week gestational age brain).
        """
        brain_volume, _ = self.brain_volumne_computation(segm, segm_affine)
        max_ratio = brain_volume/0.444 # Based on 38gw brain volume

        return max_ratio

    
    def length_ratio(self, segm, segm_affine):
        """
        Computes the scaling ratio between a reference anterior-posterior brain length 
        (from a 21-week gestational age brain) and the length of the brain in the input segmentation.

        Args:
            segm (torch.Tensor): 3D tensor representing the brain segmentation.
            segm_affine (torch.Tensor): 4x4 affine matrix associated with the segmentation, 
                used to determine voxel spacing.

        Returns:
            length_ratio (float): The ratio of the reference brain length (80 voxels, typical for 
                21-week gestational age) to the measured brain length in the Y-axis from the input segmentation.
        """
        _, brain_y_length = self.brain_volumne_computation(segm, segm_affine)
        length_ratio = 80/brain_y_length # Based on 21gw brain length

        return length_ratio

    def validate_range(self, attr_min: str, attr_max: str, min_default=1, max_default=1):
        """Ensures that min/max attributes are valid for random selection and processing."""
        min_val = getattr(self, attr_min)
        max_val = getattr(self, attr_max)

        # Ensure min > 0, struct of size 0 returns error
        if min_val == 0:
            min_val = min_default
        if max_val == 0:
            max_val = max_default

        # Swap if out of order
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        # Save corrected values back
        setattr(self, attr_min, min_val)
        setattr(self, attr_max, max_val)


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
            NN_tensor (torch.Tensor): Interpolated tensor where missing values are filled.
        """
        if dilated_mask is not None:
            # Seed Interpolation Mode: Find missing values only in the dilated mask
            NN_tensor = mask.clone()
            missing_values = torch.nonzero((dilated_mask == 1) & (mask == 0))
            # Reference dataset for the nearest neighbor search
            valid_indices = torch.nonzero(mask)
        else:
            # General Interpolation Mode: Work on the entire tensor
            NN_tensor = tensor.clone()
            NN_tensor[mask > 0.5] = 0  # Remove mask label
            missing_values = torch.nonzero(mask)
            # Reference dataset for the nearest neighbor search
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
        """Extract the biggest connected component from a binary mask and apply nearest
        neighbor interpolation.

        Args:
            target_mask (torch.Tensor): Binary mask representing the region of interest.
            tensor (torch.Tensor): Segmentation tensor to be modified.

        Returns:
            component_mask (torch.Tensor): Mask of the largest connected component.
            tensor (torch.Tensor): Updated segmentation tensor with the largest component preserved.
        """
        # For 26-connectivity
        structure = np.ones((3,3,3), dtype=int)
        labeled_mask, _ = label(self.tensor_to_numpy(target_mask), structure=structure)
        labeled_mask = self.numpy_to_tensor(labeled_mask).to(torch.long)

        component_sizes = torch.bincount(labeled_mask.view(-1))
        # If only background, no CC, skip the cc alterations
        if len(component_sizes) < 2: return target_mask, tensor, True
        sorted_indices = torch.argsort(component_sizes[1:], descending=True) + 1
        target_label = sorted_indices[0]
        component_mask = labeled_mask == target_label

        tensor = self.NN_interpolation(tensor, target_mask)
        tensor[component_mask] = self.target_label

        return component_mask, tensor, False
    

    def random_mask(self, target_mask, segmentation, NN_tensor, NN_seed, seed, genparams):
        """
        Generate a random mask transformation based on three options: original mask, hyperplasia,
        or hypoplasia. Only compute eroded or dilated masks if they are selected.

        Args:
            target_mask (torch.Tensor): Binary mask indicating the affected area.
            segmentation (torch.Tensor): Segmentation tensor.
            NN_tensor (torch.Tensor): Interpolated segmentation tensor.
            NN_seed (torch.Tensor): Seed tensor after interpolation.
            seed (torch.Tensor): Original seed tensor.
            genparams (dict): Dictionary containing transformation parameters.

        Returns:
            selected_target_mask (torch.Tensor): Transformed target mask after
                applying random pathology.
        """

        probabilities = torch.tensor([0.7, 0.1, 0.2])
        selected_index = torch.multinomial(probabilities, 1).item()

        if selected_index == 0:
            # Hypoplasia (eroded mask)
            selected_target_mask, _, _, _ = self.hypoplasia(target_mask, NN_tensor, NN_seed, seed, genparams.copy())
        elif selected_index == 1:
            # Hyperplasia (dilated mask)
            selected_target_mask, _, _, _ = self.hyperplasia(target_mask, segmentation, seed, genparams.copy())
        else:
            # Original mask
            selected_target_mask = target_mask
            
        return selected_target_mask


    def random_alteration(self, seed, seed_csf, segmentation, genparams: dict = {}):
        """
        Applies a random or specified cc anaotmy alteration to a segmentation mask and its corresponding seed.

        This function introduces structural abnormalities to a segmentation by simulating various cc alterations such as 
        hyperplasia, hypoplasia, partial loss, kinked deformation, or agenesis. The transformation is conditionally applied 
        based on a random probability threshold. If a connected component exists in the target label, it is isolated and used 
        as the base for the transformation. If `genparams` are provided, they guide the choice of alteration; otherwise, a 
        transformation is randomly selected using predefined probabilities.

        The function also supports the application of additional brain-wide alterations through the `brain_alteration` object 
        if it is defined.

        Args:
            seed (torch.Tensor): The binary mask (seed) corresponding to the target structure before any transformation.
            seed_csf (torch.Tensor): A variant of the seed where the target structure is replaced by CSF, used for interpolation.
            segmentation (torch.Tensor): The full segmentation mask that includes the target label to be altered.
            genparams (dict, optional): Parameters specifying which pathological alteration to apply. If empty, a random
                alteration is selected. Keys can include:
                - 'kinked_amplitude', 'kinked_freq'
                - 'size_dilation'
                - 'size_erosion'
                - 'anterior_loss', 'posterior_loss'
                - 'angenesis'

        Returns:
            torch.Tensor: Transformed segmentation mask after applying the selected alteration.
            torch.Tensor: Transformed seed mask corresponding to the altered region.
            dict: Updated generation parameters reflecting the applied transformation.
        """

        self.alteration_prob = random.random()
        if self.alteration_prob > self.cc_threshold:
            target_mask = segmentation == self.target_label
            target_mask, segmentation, indicator = self.biggest_connected_component(target_mask, segmentation)
            
            # If only background, no CC, skip the cc alterations
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
                # For the base images, cc is substituted by csf
                NN_segm = segmentation.clone()
                NN_segm[target_mask] = self.csf_label
                #NN_segm = self.NN_interpolation(segmentation, target_mask)
                #NN_seed = self.NN_interpolation(seed, target_mask)
                NN_seed = seed_csf.clone()
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
                    # Kinked and partial_loss alterations have a ranodom mask between original, eroded or dilated
                    elif selected_transformation == self.kinked:
                        _, transformed_segmentation, transformed_seed, genparams = selected_transformation(selected_target_mask, segmentation, seed, genparams)
                    else:
                        _, transformed_segmentation, transformed_seed, genparams = selected_transformation(selected_target_mask, NN_segm, NN_seed, seed, genparams)
                
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
            dilated_mask (torch.Tensor): Updated mask after dilation.
            dilated_tensor (torch.Tensor): Updated segmentation tensor after dilation.
            dilated_seed (torch.Tensor): Updated seed tensor after dilation.
            genparams (dict): Updated transformation parameters.
        """
        # Defined relative to brain volume
        self.max_dilation = round(self.max_dilation * self.max_ratio(segm, segm.affine))
        self.validate_range('min_dilation', 'max_dilation')
        size = genparams.get('size_dilation', random.randint(self.min_dilation, self.max_dilation))
        genparams['size_dilation'] = size
        # Base segmentation image
        dilated_tensor = segm.clone()
        # Base seed image
        dilated_seed = seed.clone()

        NN_mask = torch.zeros_like(seed, dtype=torch.int64)
        NN_mask[target_mask] = seed[target_mask]

        struct = torch.tensor(ball(size))
        dilated_mask = torch.tensor(binary_dilation(self.tensor_to_numpy(target_mask), structure=self.tensor_to_numpy(struct)))

        # Use NN_interpolation to obtain the extended seed mask
        extended_mask = self.NN_interpolation(seed, NN_mask, dilated_mask) 

        # Reinsert extended mask on top of base seed image
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
            eroded_mask (torch.Tensor): Updated mask after erosion.
            eroded_tensor (torch.Tensor): Updated segmentation tensor after erosion.
            eroded_seed (torch.Tensor):  Updated seed segmentation tensor after erosion.
            genparams (dict): Updated generation parameters.
        """
        self.validate_range('min_erosion', 'max_erosion')
        size = genparams.get('size_erosion', random.randint(self.min_erosion, self.max_erosion))
        genparams['size_erosion'] = size

        struct = torch.ones((size, 1), dtype=torch.int).unsqueeze(0)
        eroded_mask = torch.tensor(binary_erosion(self.tensor_to_numpy(target_mask), structure=self.tensor_to_numpy(struct)))

        # Post-processing step: only keep the biggest connected component after erosion
        eroded_mask, NN_segm, _ = self.biggest_connected_component(eroded_mask, NN_segm)

        eroded_seed = NN_seed.clone()
        # Reinsert dilated mask on top of base seed image
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
            genparams (dict): Updated generation parameters.
            mask_condition (torch.Tensor): Boolean mask indicating the affected area.
        """
        self.validate_range('min_posterior_loss', 'max_posterior_loss', min_default=0, max_default=0)
        percentage = genparams.get('posterior_loss', random.uniform(self.min_posterior_loss, self.max_posterior_loss))
        genparams['posterior_loss'] = percentage

        segment = min_y + int(percentage * total_length)
        mask_condition = (coords[:, 1] > segment)

        return genparams, mask_condition
    

    def anterior_loss(self, min_y, total_length, coords, genparams):
        """Apply anterior loss by removing a random percentage of the anterior region.

        Args:
            min_y (int): Minimum y-coordinate of the affected region.
            total_length (int): Total extent of the affected region.
            coords (torch.Tensor): Coordinates of the affected region.
            genparams (dict): Dictionary of generation parameters.

        Returns:
            genparams (dict): Updated generation parameters.
            mask_condition (torch.Tensor): Boolean mask indicating the affected area.
        """
        self.validate_range('min_anterior_loss', 'max_anterior_loss', min_default=0, max_default=0)
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
            partial_loss_mask (torch.Tensor): Updated mask indicating the removed region.
            partial_loss_segm (torch.Tensor): Updated segmentation after applying the loss.
            partial_loss_seed (torch.Tensor): Updated seed segmentation after applying the loss.
            genparams (dict): Updated generation parameters.
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
        # Reinsert partial mask on top of base seed image
        partial_loss_seed[partial_loss_mask] = seed[partial_loss_mask]
        partial_loss_segm = NN_segm.clone()
        partial_loss_segm[partial_loss_mask] = self.target_label

        return partial_loss_mask, partial_loss_segm, partial_loss_seed, genparams


    def kinked(self, target_mask, segm, seed, genparams):
        """Apply a sinusoidal deformation to the specified region of a segmentation mask, producing a 
        "kinked" distortion effect by warping the x-coordinates based on the y-coordinate.

        This transformation simulates structural irregularities by introducing a smooth, wave-like 
        deviation along the x-axis within a region of interest (ROI) defined by `target_mask`.
        The deformation is parameterized by amplitude and frequency, which can be controlled via 
        `genparams` or randomly sampled within configured bounds.

        Args:
            target_mask (torch.Tensor): Binary mask of the region to modify.
            segm (torch.Tensor): Segmentation tensor.
            seed (torch.Tensor): Seed segmentation tensor. 
            genparams (dict): Dictionary of generation parameters.

        Returns:
            None: Placeholder return value.
            kinked_segm (torch.Tensor): Updated segmentation after the kinked transformation.
            kinked_seed (torch.Tensor): Updated seed segmentation after transformation.
            genparams (dict): Updated generation parameters.
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
            self.validate_range('min_amplitude_kinked', 'max_amplitude_kinked', min_default=0, max_default=0)
            amplitude = genparams.get('kinked_amplitude', random.uniform(self.min_amplitude_kinked, self.max_amplitude_kinked))
            self.validate_range('min_freq_kinked', 'max_freq_kinked', min_default=0, max_default=0)
            freq = genparams.get('kinked_freq', random.uniform(self.min_freq_kinked, self.max_freq_kinked))
            # Convert frequency into radians 
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
            None: Placeholder return value.
            NN_segm (torch.Tensor): The nearest-neighbor interpolated segmentation remains unchanged.
            NN_seed (torch.Tensor): The nearest-neighbor interpolated seed segmentation remains unchanged.
            genparams (dict): Updated generation parameters with agenesis set to 1, as an boolean indicator that 
                agenesis transformation has been applied.
        """
        genparams['agenesis'] = 1
        
        return None, NN_segm, NN_seed, genparams