import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from collections import defaultdict

class PAFocusedDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, 
                 pa_threshold=0.001, pa_slice_ratio=0.7):
        """
        Enhanced dataset for PA segmentation with focus on PA-containing slices
        but using whole images instead of patches
        
        Args:
            base_dir: Base directory for data
            list_dir: Directory with train/val/test lists
            split: 'train', 'val', or 'test'
            transform: Transforms to apply
            pa_threshold: Minimum PA percentage to consider a slice as "positive"
            pa_slice_ratio: Ratio of positive slices to include in training
        """
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.pa_threshold = pa_threshold
        self.pa_slice_ratio = pa_slice_ratio
        self.cache = {}
        
        # For training, we'll pre-scan the data to find positive slices
        if self.split == "train":
            self._scan_positive_slices()
        
    def _scan_positive_slices(self):
        """
        Pre-scan the dataset to identify slices containing PA
        """
        print("Scanning dataset for PA-containing slices...")
        self.positive_slices = defaultdict(list)
        self.case_slices = {}
        
        for idx, case_line in enumerate(self.sample_list):
            case_name = case_line.strip('\n')
            
            # Extract case name (handle _slice suffix)
            if "_slice" in case_name:
                pa_id, _ = case_name.split('_slice')
            else:
                pa_id = case_name
            
            # Skip if we've already processed this case
            if pa_id in self.case_slices:
                continue
                
            # Path to the label file
            lab_path = os.path.join(self.data_dir, pa_id, "label", f"{pa_id}.nii.gz")
            
            # Read the label file
            lab_sitk = sitk.ReadImage(lab_path)
            lab_data = sitk.GetArrayFromImage(lab_sitk)
            
            # Store the total number of slices for this case
            total_slices = lab_data.shape[0]
            self.case_slices[pa_id] = total_slices
            
            # Find slices with significant PA content
            for slice_idx in range(total_slices):
                slice_data = lab_data[slice_idx]
                slice_pa_ratio = np.sum(slice_data > 0) / slice_data.size
                
                # If the slice has significant PA content, add it to positive slices
                if slice_pa_ratio >= self.pa_threshold:
                    self.positive_slices[pa_id].append(slice_idx)
        
        # Count total positive slices
        total_positive = sum(len(slices) for slices in self.positive_slices.values())
        print(f"Found {total_positive} slices with PA content (≥{self.pa_threshold*100:.3f}%)")
        
        # Create a sampling list that weights positive slices
        self.sampling_list = []
        
        # Add all positive slices
        for pa_id, slice_indices in self.positive_slices.items():
            for slice_idx in slice_indices:
                self.sampling_list.append((pa_id, slice_idx, True))  # (case_id, slice_idx, is_positive)
        
        # Add some negative slices (to maintain balance)
        for pa_id, total_slices in self.case_slices.items():
            positive_indices = set(self.positive_slices[pa_id])
            # Calculate how many negative slices to add
            num_positive = len(positive_indices)
            # Add enough negative slices to maintain the desired ratio
            num_negative_to_add = int(num_positive * (1 - self.pa_slice_ratio) / self.pa_slice_ratio)
            
            # Get list of negative slices
            negative_indices = [i for i in range(total_slices) if i not in positive_indices]
            
            # Randomly sample negative slices
            if num_negative_to_add > 0 and len(negative_indices) > 0:
                # Don't sample more than available
                num_negative_to_add = min(num_negative_to_add, len(negative_indices))
                sampled_negative = np.random.choice(negative_indices, num_negative_to_add, replace=False)
                
                for slice_idx in sampled_negative:
                    self.sampling_list.append((pa_id, slice_idx, False))
        
        print(f"Created sampling list with {len(self.sampling_list)} entries")
        print(f"Positive slices: {total_positive}, Total in sampling: {len(self.sampling_list)}")
        
        # Shuffle the sampling list
        np.random.shuffle(self.sampling_list)

    def __len__(self):
        if self.split == "train":
            return len(self.sampling_list)
        else:
            return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            # Use our weighted sampling strategy
            pa_id, slice_num, is_positive = self.sampling_list[idx]
            case_name = f"{pa_id}_slice{slice_num}"
        else:
            # For validation/testing, use the original list
            case_name = self.sample_list[idx].strip('\n')
            
            # Extract case name and slice if applicable
            if "_slice" in case_name:
                pa_id, slice_id = case_name.split('_slice')
                slice_num = int(slice_id)
            else:
                pa_id = case_name
                slice_num = 0
        
        # Path to the NIfTI files
        img_path = os.path.join(self.data_dir, pa_id, "image", f"{pa_id}.nii.gz")
        lab_path = os.path.join(self.data_dir, pa_id, "label", f"{pa_id}.nii.gz")
        
        # Check if already cached
        cache_key = f"{img_path}_{slice_num}"
        if cache_key in self.cache:
            img_slice, lab_slice = self.cache[cache_key]
        else:
            # Read data directly using SimpleITK
            img_sitk = sitk.ReadImage(img_path)
            img_data = sitk.GetArrayFromImage(img_sitk)
            
            lab_sitk = sitk.ReadImage(lab_path)
            lab_data = sitk.GetArrayFromImage(lab_sitk)
            
            # Extract specific slice
            img_slice = img_data[slice_num].astype(np.float32)
            lab_slice = lab_data[slice_num].astype(np.uint8)
            
            # Normalize image
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            
            # Ensure binary labels
            lab_slice[lab_slice > 0] = 1
            
            # Cache for future use
            self.cache[cache_key] = (img_slice, lab_slice)
        
        sample = {'image': img_slice, 'label': lab_slice}
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = case_name
        return sample