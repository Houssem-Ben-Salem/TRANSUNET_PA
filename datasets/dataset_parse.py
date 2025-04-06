import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Parse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.cache = {}  # Optional cache for better performance

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n')
        
        if self.split == "train":
            # For training data, the filename includes slice information
            if "_slice" in case_name:
                pa_id, slice_id = case_name.split('_slice')
                slice_num = int(slice_id)
            else:
                pa_id = case_name
                slice_num = 0  # Default slice
            
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
                
                # Cache for future use (use with caution for large datasets)
                self.cache[cache_key] = (img_slice, lab_slice)
            
        else:  # Testing
            # For test volumes, load the entire 3D volume
            img_path = os.path.join(self.data_dir, case_name, "image", f"{case_name}.nii.gz")
            lab_path = os.path.join(self.data_dir, case_name, "label", f"{case_name}.nii.gz")
            
            img_sitk = sitk.ReadImage(img_path)
            img_data = sitk.GetArrayFromImage(img_sitk)
            
            lab_sitk = sitk.ReadImage(lab_path)
            lab_data = sitk.GetArrayFromImage(lab_sitk)
            
            # Normalize each slice
            img_slice = img_data.astype(np.float32)
            for i in range(img_slice.shape[0]):
                slice_i = img_slice[i]
                if slice_i.max() > slice_i.min():  # Avoid division by zero
                    img_slice[i] = (slice_i - slice_i.min()) / (slice_i.max() - slice_i.min())
            
            # Ensure binary labels
            lab_slice = lab_data.astype(np.uint8)
            lab_slice[lab_slice > 0] = 1
        
        sample = {'image': img_slice, 'label': lab_slice}
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = case_name
        return sample