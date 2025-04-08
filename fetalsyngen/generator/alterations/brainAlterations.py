#from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.neighbors import NearestNeighbors  # Intel optimized NearestNeighbors
import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import ball
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from sklearn.neighbors import NearestNeighbors
import random
import nibabel as nib
import skfmm
import time


class brain_Alterations:
    def __init__(self, alteration_prob: float, cortex_label: int,
                 csf_label: int, wm_label: int, ventri_label: int, 
                 cerebellum_label: int, deep_gm_label: int, 
                 brainstem_label: int, cc_label: int,
                 min_size_smooth: int, max_size_smooth: int,
                 min_iter_smooth: int, max_iter_smooth: int,
                 min_size_thick: int, max_size_thick:int,
                 min_size_thin: int,
                 ventri_max_strength: float, ventri_min_strength: float,
                 ventri_max_radius: float, ventri_min_radius: float,
                 left_ventri: int, right_ventri: int, bilateral_ventri: int,
                 min_size_hypo: int, brain_alter_prob: torch.Tensor

):
        """Initialize brain alteration parameters.
        
        Args:
            alteration_prob (float): Probability of applying an alteration.
            cortex_label (int): Label for cortical regions.
            csf_label (int): Label for cerebrospinal fluid (CSF).
            wm_label (int): Label for white matter.
            ventri_label (int): Label for ventricular regions.
            cerebellum_label (int): Label for cerebellum.
            deep_gm_label (int): Label for deep gray matter.
            brainstem_label (int): Label for brainstem.
            cc_label (int): Label for corpus callosum.
            min_size_smooth (int), max_size_smooth (int): Min/max kernel sizes for smoothing.
            min_iter_smooth (int), max_iter_smooth (int): Min/max iterations for smoothing.
            min_size_thick (int), max_size_thick (int): Min/max kernel sizes for thickening.
            min_size_thin (int), max_size_thin (int): Min/max kernel sizes for thinning.
            ventri_max_strength (float), ventri_min_strength (float): Strength range for ventriculomegaly.
            ventri_max_radius (float), ventri_min_radius (float): Radius range for ventriculomegaly.
            left_ventri (int), right_ventri (int), bilateral_ventri (int): Ventricular region flags.
            min_size_hypo (int), max_size_hypo (int): Min/max kernel sizes for hypoplasia.
        """
        self.alteration_prob = alteration_prob
        self.cortex_label = cortex_label
        self.csf_label = csf_label
        self.wm_label = wm_label
        self.ventri_label = ventri_label
        self.cerebellum_label = cerebellum_label
        self.deep_gm_label = deep_gm_label
        self.brainsetm_label = brainstem_label
        self.cc_label = cc_label
        self.min_size_smooth = min_size_smooth
        self.max_size_smooth = max_size_smooth
        self.min_iter_smooth = min_iter_smooth
        self.max_iter_smooth = max_iter_smooth
        self.min_size_thick = min_size_thick
        self.max_size_thick = max_size_thick
        self.min_size_thin = min_size_thin
        self.ventri_max_strength = ventri_max_strength
        self.ventri_min_strength = ventri_min_strength
        self.ventri_max_radius = ventri_max_radius
        self.ventri_min_radius = ventri_min_radius
        self.left_ventri = left_ventri
        self.right_ventri = right_ventri
        self.bilateral_ventri = bilateral_ventri
        self.min_size_hypo = min_size_hypo
        self.brain_alter_prob = torch.tensor(brain_alter_prob, dtype=torch.float32)


    def numpy_to_tensor(self, array):
        """Convert a NumPy array to a PyTorch tensor."""
        return torch.tensor(array, dtype=torch.int64)


    def tensor_to_numpy(self, tensor):
        """Convert a PyTorch tensor to a NumPy array."""
        return tensor.cpu().numpy()
    
    def brain_volumne_computation(self, segm, segm_affine):
        """Computes brain volume in dm3."""
        voxel_size = torch.abs(torch.linalg.det(segm_affine[:3, :3]))
        # Automatically get all non-background labels
        unique_labels = torch.unique(segm).int()
        brain_tissues = unique_labels[unique_labels != 0]
        brain_mask = torch.isin(segm, brain_tissues)

        brain_voxels = brain_mask.sum()
        brain_volume_mm3 = brain_voxels * voxel_size
        brain_volume_dm3 = float(brain_volume_mm3 / 1000000)

        return brain_volume_dm3
    
    
    def max_ratio(self, segm, segm_affine):
        """Computes a scaling ratio based on a reference 38-week gestational age (GW) brain volume atlas."""
        brain_volume = self.brain_volumne_computation(segm, segm_affine)
        max_ratio = brain_volume/0.257 # Based on 38gw atlas brain volume

        return max_ratio

    def thin_range(self, segm, segm_affine):
        """Determines the threshold for thinning operations based on brain volume."""
        brain_volume = self.brain_volumne_computation(segm, segm_affine)
        if brain_volume > 0.3: return 2 # Above 31gw
        else: return 1

    def hypo_range(self, segm, segm_affine):
        """ Determines the threshold for hypoplasia severity based on brain volume."""
        brain_volume = self.brain_volumne_computation(segm, segm_affine)
        if brain_volume > 0.2: return 3 # Above 26gw
        else: return 2

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
    

    def create_pinch_grid_3d(self, d, h, w, strength, radius, center):
        """Create a pinch grid for 3D images.

        Args:
            d, h, w: The depth, height, and width of the grid.
            strength: The strength of the pinch effect. A value of 0 will have no effect,
                while a value of 1 will completely pinch the image to the center.
            radius: The radius of the pinch effect. If None, the radius will be the smallest
                value that fits within the grid.
            center: The center of the pinch effect. If None, the center will be the middle
                of the grid. Should be a tuple of normalized coordinates (x, y, z).

        Returns:
            A tensor with shape (D, H, W, 3) containing the pinch grid.
        """
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        grid = torch.stack((x, y, z), dim=-1)

        if center is None:
            center = (0.0, 0.0, 0.0) 
        center_z, center_y, center_x = center

        if radius is None:
            radius = 1.0 

        # Compute distances from the center
        dx = grid[..., 0] - center_x
        dy = grid[..., 1] - center_y
        dz = grid[..., 2] - center_z
        distance = torch.sqrt(dx**2 + dy**2 + dz**2)

        # Compute the pinch factor
        factor = torch.ones_like(distance)
        mask = distance < radius
        factor[mask] = 1 - strength * ((radius - distance[mask]) / radius) ** 2

        # Apply the pinch effect to the grid
        grid[..., 0] = center_x + dx * factor
        grid[..., 1] = center_y + dy * factor
        grid[..., 2] = center_z + dz * factor

        return grid



    def random_brain_alteration(self, seed, segmentation, genparams: dict = {}):
        """
        Apply a random brain alteration transformation to the given brain segmentation 
        based on predefined probabilities and input parameters.

        Parameters:
        - seed: A random seed for reproducibility.
        - segmentation: Tensor representing the segmented brain regions.
        - genparams (dict): Dictionary containing parameters for specific transformations.

        Returns:
        - transformed_segmentation: The updated segmentation tensor after alteration.
        - transformed_seed: The uodated seed value after alteration.
        - genparams: Updated dictionary of parameters.
        """
        if self.alteration_prob > 0.5:
            cortex_mask = segmentation == self.cortex_label
            csf_mask = segmentation == self.csf_label
            ventricles_mask = segmentation.clone()
            ventricles_mask[ventricles_mask != self.ventri_label] = 0

            if 'size_smooth' in genparams or 'iter_smooth' in genparams:
                transformed_segmentation, transformed_seed, genparams = self.smoother_cortex(cortex_mask, segmentation, seed, genparams)
            elif 'size_thick' in genparams:
                transformed_segmentation, transformed_seed, genparams = self.thicker_cortex(cortex_mask, segmentation, seed, genparams)
            elif 'size_thin' in genparams:
                transformed_segmentation, transformed_seed, genparams = self. thinner_cortex(csf_mask, segmentation, seed, genparams)
            elif 'ventri_left_strength' in genparams or 'ventri_right_strength' in genparams:
                transformed_segmentation, transformed_seed, genparams = self.ventriculomegaly(ventricles_mask, segmentation, seed, genparams)
            elif 'size_hypo' in genparams:
                transformed_segmentation, transformed_seed, genparams = self.hypoplasia(segmentation, seed, genparams)
            else:
                pathology_functions = [
                    self.smoother_cortex,
                    self.thicker_cortex,
                    self.thinner_cortex,
                    self.ventriculomegaly,
                    self.hypoplasia
                ]
                selected_index = torch.multinomial(self.brain_alter_prob, 1).item()
                selected_transformation = pathology_functions[selected_index]
                
                if selected_transformation == self.thinner_cortex:
                    transformed_segmentation, transformed_seed, genparams = selected_transformation(csf_mask, segmentation, seed, genparams) 
                elif selected_transformation == self.ventriculomegaly:
                    transformed_segmentation, transformed_seed, genparams = selected_transformation(ventricles_mask, segmentation, seed, genparams)
                elif selected_transformation == self.hypoplasia:
                    transformed_segmentation, transformed_seed, genparams = selected_transformation(segmentation, seed, genparams)
                else: 
                    transformed_segmentation, transformed_seed, genparams = selected_transformation(cortex_mask, segmentation, seed, genparams) 
            return transformed_segmentation, transformed_seed, genparams
            
        else:
            return segmentation, seed, {}


    def smoother_cortex(self, cortex_mask, segm, seed, genparams):
        """
        Apply a smoothing transformation by dilating and then eroding the cortex mask.

        Parameters:
        - cortex_mask: A boolean mask indicating the cortex region in the segmentation.
        - segm: The original brain segmentation.
        - seed: The original seed tensor.
        - genparams:  Dictionary containing transformation parameters.

        Returns:
        - smooth_cortex_segm: The modified brain segmentation after transformation.
        - smooth_cortex_seed: The modified seed tensor after transformation.
        - genparams: Updated generation parameters.
        """
        size = genparams.get('size_smooth', random.randint(self.min_size_smooth, self.max_size_smooth))
        genparams['size_smooth'] = size
        iter = genparams.get('iter_smooth', random.randint(self.min_iter_smooth, self.max_iter_smooth))
        genparams['iter_smooth'] = iter

        NN_seed = torch.zeros_like(seed, dtype=torch.int64)
        NN_seed[cortex_mask] = seed[cortex_mask]

        struct = torch.tensor(ball(size))
        smooth_cortex_mask = cortex_mask.clone() 
        smooth_cortex_mask = torch.tensor(binary_dilation(self.tensor_to_numpy(smooth_cortex_mask), structure=self.tensor_to_numpy(struct), iterations=iter))
        smooth_cortex_mask = torch.tensor(binary_erosion(self.tensor_to_numpy(smooth_cortex_mask), structure=self.tensor_to_numpy(struct), iterations=iter))
        extended_mask = self.NN_interpolation(seed, NN_seed, smooth_cortex_mask) 

        smooth_cortex_seed = seed.clone()
        smooth_cortex_segm = segm.clone()
        # Only extentd on the csf, do not eat wm
        mask_condition = torch.zeros_like(seed, dtype=torch.int64)
        mask_condition = (smooth_cortex_mask) & (smooth_cortex_segm == self.csf_label)
        
        smooth_cortex_seed[mask_condition] = extended_mask[mask_condition]
        smooth_cortex_segm[mask_condition] = self.cortex_label

        return smooth_cortex_segm, smooth_cortex_seed, genparams
    

    def thicker_cortex(self, cortex_mask, segm, seed, genparams):
        """
        Apply hyperplasia transformation by dilating the affected region, specifically to the cortex (gray matter).
        
        Parameters:
        - cortex_mask: A boolean mask indicating the cortex region in the segmentation.
        - segm: The original brain segmentation.
        - seed: The original seed tensor.
        - genparams: A dictionary containing transformation parameters.

        Returns:
        - thick_cortex_segm: The modified brain segmentation after transformation (with thicker cortex).
        - thick_cortex_seed: The modified seed tensor after transformation.
        - genparams: Updated generation parameters.
        """
        size = genparams.get('size_thick', random.randint(self.min_size_thick, self.max_size_thick))
        genparams['size_thick'] = size

        NN_seed = torch.zeros_like(seed, dtype=torch.int64)
        NN_seed[cortex_mask] = seed[cortex_mask]

        struct = torch.tensor(ball(size))
        thick_cortex_mask = torch.tensor(binary_dilation(self.tensor_to_numpy(cortex_mask), structure=self.tensor_to_numpy(struct)))
        extended_mask = self.NN_interpolation(seed, NN_seed, thick_cortex_mask) 

        thick_cortex_seed = seed.clone()
        thick_cortex_segm = segm.clone()
        # Only extentd the gm (cortex) on the wm, csf remains intact
        mask_condition = torch.zeros_like(seed, dtype=torch.int64)
        mask_condition = (thick_cortex_mask) & (thick_cortex_segm == self.wm_label)

        thick_cortex_seed[mask_condition] = extended_mask[mask_condition]
        thick_cortex_segm[mask_condition] = self.cortex_label

        return thick_cortex_segm, thick_cortex_seed, genparams
    

    def thinner_cortex(self, csf_mask, segm, seed, genparams):
        """
        Apply hypoplasia transformation by dilating the CSF region to erode the cortex (gray matter).

        Parameters:
        - csf_mask: A boolean mask indicating the CSF region in the segmentation.
        - segm: The original brain segmentation.
        - seed: The original seed tensor.
        - genparams: A dictionary containing transformation parameters.

        Returns:
        - thin_cortex_segm: The modified brain segmentation after transformation (with thinner cortex).
        - thin_cortex_seed: The modified seed tensor after transformation.
        - genparams: Updated generation parameters.
        """
        max_size_thin = self.thin_range(segm, segm.affine)
        if self.min_size_thin == max_size_thin: size = max_size_thin
        else:
            size = genparams.get('size_thin', random.randint(self.min_size_thin, max_size_thin))
        genparams['size_thin'] = size

        NN_seed = torch.zeros_like(seed, dtype=torch.int64)
        NN_seed[csf_mask] = seed[csf_mask]

        struct = torch.tensor(ball(size))
            
        thin_cortex_mask = torch.tensor(binary_dilation(self.tensor_to_numpy(csf_mask), structure=self.tensor_to_numpy(struct)))

        extended_mask = self.NN_interpolation(seed, NN_seed, thin_cortex_mask)

        thin_cortex_seed = seed.clone()
        thin_cortex_segm = segm.clone()
        # Only extentd the csf on the gm (crotex) to erode it
        mask_condition = torch.zeros_like(seed, dtype=torch.int64)
        mask_condition = (thin_cortex_mask) & (thin_cortex_segm == self.cortex_label)
       
        thin_cortex_seed[mask_condition] = extended_mask[mask_condition]
        thin_cortex_segm[mask_condition] = self.csf_label

        return thin_cortex_segm, thin_cortex_seed, genparams
    

    def apply_ventriculomegaly(self, segm, seed, ventricle_mask, strength, radius):
        """
        Simulates ventriculomegaly (enlargement of the ventricles) by applying a "pinching" transformation 
        to the given tensor (segmentation) and seed tensor, focusing on randomly selected points within 
        the ventricle mask (either left or right).

        Parameters:
        - segm: The original brain segmentation.
        - seed: The original seed tensor.
        - ventricle_mask: A binary mask identifying the ventricle regions in the brain (either left or right).
        - strength: The strength of the ventriculomegaly effect (how much the ventricles should be enlarged).
        - radius: The radius of influence for the ventriculomegaly effect (how far the transformation should spread).

        Returns:
        - pinched_segm: The modified segmentation tensor after applying the ventriculomegaly transformation.
        - pinched_seed: The modified seed tensor after the transformation.
        """
        segm = segm.unsqueeze(0).unsqueeze(0).float()
        seed = seed.unsqueeze(0).unsqueeze(0).float()
        d, h, w = segm.shape[2:]

        coords = torch.where(ventricle_mask[0, 0] != 0)
        idx = np.random.randint(coords[0].shape[0])
        random_point = coords[0][idx], coords[1][idx], coords[2][idx]
        norm_random_point = random_point / np.array([d, h, w]) * 2 - 1  # Normalized coordinates

        pinch_grid = self.create_pinch_grid_3d(d, h, w, strength, radius, norm_random_point)
        pinch_grid = pinch_grid.unsqueeze(0)
        pinched_segm = F.grid_sample(segm, pinch_grid, mode="nearest", padding_mode="border", align_corners=True)
        pinched_seed = F.grid_sample(seed, pinch_grid, mode="nearest", padding_mode="border", align_corners=True)
        pinched_segm = pinched_segm.squeeze(0).squeeze(0).long()
        pinched_seed = pinched_seed.squeeze(0).squeeze(0).long()

        return  pinched_segm, pinched_seed, ventricle_mask
    

    def left_ventricle(self, ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams):
        """
        Preparates left ventricle arguments to apply a ventriculomegaly transformation to the left ventricle region 
        of the brain segmentation.

        Parameters:
        - ventriculomegaly_segm: The brain segmentation tensor.
        - ventriculomegaly_seed: The seed tensor.
        - inside_mask_np: A binary mask representing the inside points of the left ventricle from where to chose a random point.
        - midsagittal_x: The x-coordinate of the midsagittal plane (used to split the brain into left and right hemispheres).
        - genparams: A dictionary containing the generation parameters for the transformation.

        Returns:
        - ventriculomegaly_segm: The modified brain segmentation after applying the left ventriculomegaly transformation.
        - ventriculomegaly_seed: The modified seed tensor after applying the transformation.
        """
        inside_mask_left = np.zeros_like(inside_mask_np)
        inside_mask_left[:midsagittal_x, :, :] = inside_mask_np[:midsagittal_x, :, :]

        inside_mask_left_segm = torch.tensor(inside_mask_left, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        left_ventri_strength = genparams.get('left_ventri_strength', random.uniform(self.ventri_min_strength, self.ventri_max_strength))
        genparams['left_ventri_strength'] = left_ventri_strength
        left_ventri_radius = genparams.get('left_ventri_radius', random.uniform(self.ventri_min_radius, self.ventri_max_radius))
        genparams['left_ventri_radius'] = left_ventri_radius
        
        ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_left_segm = self.apply_ventriculomegaly(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_left_segm, left_ventri_strength, left_ventri_radius)

        return ventriculomegaly_segm, ventriculomegaly_seed
    
    
    def right_ventricle(self, ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams):
        """
        Preparates right ventricle arguments to apply a ventriculomegaly transformation to the right ventricle region 
        of the brain segmentation.

        Parameters:
        - ventriculomegaly_segm: The brain segmentation tensor.
        - ventriculomegaly_seed: The seed tensor.
        - inside_mask_np: A binary mask representing the inside points of the right ventricle from where to chose a random point.
        - midsagittal_x: The x-coordinate of the midsagittal plane (used to split the brain into left and right hemispheres).
        - genparams: A dictionary containing the generation parameters for the transformation.

        Returns:
        - ventriculomegaly_segm: The modified brain segmentation after applying the right ventriculomegaly transformation.
        - ventriculomegaly_seed: The modified seed tensor after applying the transformation.
        """
        inside_mask_right = np.zeros_like(inside_mask_np)
        inside_mask_right[midsagittal_x:, :, :] = inside_mask_np[midsagittal_x:, :, :]

        inside_mask_right_segm = torch.tensor(inside_mask_right, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        right_ventri_strength = genparams.get('right_ventri_strength', random.uniform(self.ventri_min_strength, self.ventri_max_strength))
        genparams['right_ventri_strength'] = right_ventri_strength
        right_ventri_radius = genparams.get('right_ventri_radius', random.uniform(self.ventri_min_radius, self.ventri_max_radius))
        genparams['right_ventri_radius'] = right_ventri_radius

        ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_right_segm = self.apply_ventriculomegaly(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_right_segm, right_ventri_strength, right_ventri_radius)

        return ventriculomegaly_segm, ventriculomegaly_seed


    def ventriculomegaly(self, ventricles_mask, segm, seed, genparams):
        """
        Apply ventriculomegaly transformation to brain ventricles (dilation of ventricles) using specified parameters.
        The transformation can be applied to either one hemisphere or both based on the provided generation parameters, in 
        an asymmetric approach.

        Parameters:
        - ventricles_mask: A binary mask that indicates the location of the ventricles.
        - segm: The original brain segmentation tensor.
        - seed: The origianl seed tensor.
        - genparams: A dictionary containing generation parameters for the transformation.

        Returns:
        - ventriculomegaly_segm: The modified brain segmentation tensor after the transformation.
        - ventriculomegaly_seed: The modified seed tensor after the transformation.
        - genparams: Updated generation parameters.
        """
        ventricles_mask_np = ventricles_mask.numpy()

        self.ventri_max_radius = self.ventri_max_radius * self.max_ratio(segm, segm.affine)
        self.ventri_min_radius = self.ventri_min_radius * self.max_ratio(segm, segm.affine)

        # Compute signed distance transform (SDT)
        sdt = distance_transform_edt(ventricles_mask_np > 0.5)
        # Create a mask that keeps only pixels > 4 voxels inside
        inside_mask = sdt > 3
        
        # Get the mask where label = 1 (CSF)
        brain_mask = (segm == self.csf_label).numpy()
        # Compute the midsagittal plane based on the y plane (second dimension) from the csf mask
        brain_indices = np.where(brain_mask)  
        midsagittal_x = np.median(brain_indices[0]).astype(int)
        # Split the inside_mask using the midsagittal y plane
        inside_mask_np = inside_mask.astype(np.uint8)

        ventriculomegaly_segm = segm.clone()
        ventriculomegaly_seed = seed.clone()

        if 'left_ventri' in genparams or 'bilateral_ventri' in genparams:
            ventriculomegaly_segm, ventriculomegaly_seed = self.left_ventricle(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams)

        elif 'right_ventri' in genparams or 'bilateral_ventri' in genparams:
            ventriculomegaly_segm, ventriculomegaly_seed = self.right_ventricle(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams)

        ventri_types = [
            'bilateral',
            'unilateral_left',
            'unilateral_right'
        ]
        probabilities = torch.tensor([0.6, 0.2, 0.2])
        selected_index = torch.multinomial(probabilities, 1).item()
        ventri = ventri_types[selected_index]
        #ventri = 'unilateral_left'

        if ventri == 'unilateral_left' or ventri == 'bilateral':
            # Left hemisphere: Keep everything left of midsagittal plane (y < midsagittal_y)
            ventriculomegaly_segm, ventriculomegaly_seed = self.left_ventricle(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams)

        if ventri == 'unilateral_right' or ventri == 'bilateral':
            # Right hemisphere: Keep everything right of midsagittal plane (y >= midsagittal_y)
            ventriculomegaly_segm, ventriculomegaly_seed = self.right_ventricle(ventriculomegaly_segm, ventriculomegaly_seed, inside_mask_np, midsagittal_x, genparams)
    
        return ventriculomegaly_segm, ventriculomegaly_seed, genparams
    

    def hypoplasia(self, segm, seed, genparams):
        """
        Apply hypoplasia transformation by eroding the brain regions of interest (brainstem and cerebellum) and modifying the segmentation.

        Parameters:
        - segm: The original brain segmentation tensor containing labeled regions.
        - seed: The originalseed tensor.
        - genparams: A dictionary containing generation parameters, including the size of the erosion for the hypoplasia effect.

        Returns:
        - hypo_segm: The modified brain segmentation tensor after applying the hypoplasia transformation.
        - hypo_seed: The modified seed tensor after applying the hypoplasia transformation.
        - genparams: Updated generation parameters.
        """
        hypo = segm.clone()

        unique_labels = torch.unique(segm).int()
        labels_to_combine = unique_labels[(unique_labels != 0) & (unique_labels != self.csf_label)]
        mask = torch.isin(hypo, labels_to_combine)

        label_map = hypo.clone()
        label_map[~mask] = 0

        max_size_hypo = self.hypo_range(segm, segm.affine)
        size = genparams.get('size_hypo', random.randint(self.min_size_hypo, max_size_hypo))
        genparams['size_hypo'] = size

        struct = torch.tensor(ball(size))
        hypo_mask = torch.tensor(binary_erosion(self.tensor_to_numpy(mask), structure=self.tensor_to_numpy(struct)))
        eroded_label_map = torch.where(hypo_mask, label_map, 0)

        labels_to_retain = torch.tensor([self.brainsetm_label, self.cerebellum_label])
        eroded_mask = torch.isin(eroded_label_map, labels_to_retain)
        eroded_label_mask = torch.where(eroded_mask, label_map, 0)

        aux_mask = torch.isin(hypo, torch.tensor([self.brainsetm_label, self.cerebellum_label]))
        hypo_seed = self.NN_interpolation(seed, aux_mask)

        hypo_segm = segm.clone()
        hypo_segm[aux_mask] = 1
        hypo_segm[eroded_mask] = eroded_label_mask[eroded_mask]
        hypo_seed[eroded_mask] = seed[eroded_mask]

        return hypo_segm, hypo_seed, genparams
