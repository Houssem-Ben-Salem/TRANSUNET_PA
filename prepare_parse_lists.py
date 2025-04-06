import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random

# Define paths
data_path = "./DATA"
list_dir = "./lists/lists_Parse"

# Function to process cases and get valid slices
def process_cases_for_slices(cases, desc):
    slices = []
    for case in tqdm(cases, desc=desc):
        img_path = os.path.join(data_path, case, "image", f"{case}.nii.gz")
        lab_path = os.path.join(data_path, case, "label", f"{case}.nii.gz")
        
        # Check if files exist
        if not os.path.exists(img_path) or not os.path.exists(lab_path):
            print(f"Warning: Missing files for {case}")
            continue
        
        # Load label data to find slices with content
        lab_sitk = sitk.ReadImage(lab_path)
        lab_data = sitk.GetArrayFromImage(lab_sitk)
        
        # Find slices with meaningful content
        valid_slices = []
        for slice_idx in range(lab_data.shape[0]):
            if np.sum(lab_data[slice_idx] > 0) > 10:  # At least 10 positive pixels
                valid_slices.append(slice_idx)
        
        if not valid_slices:
            print(f"Warning: No valid slices found for {case}")
            continue
        
        # Add each valid slice to the list
        for slice_idx in valid_slices:
            slice_name = f"{case}_slice{slice_idx}"
            slices.append(f"{slice_name}\n")
    
    return slices

# Create directory if it doesn't exist
os.makedirs(list_dir, exist_ok=True)

# Get all case folders
case_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) and f.startswith('PA')]
case_folders.sort()

# Split into train and test+val sets (80% train, 20% test+val)
random.seed(42)  # For reproducibility
random.shuffle(case_folders)
train_size = int(0.8 * len(case_folders))
train_cases = case_folders[:train_size]
test_val_cases = case_folders[train_size:]

# Split the test_val set into test and validation (10% each)
val_size = len(test_val_cases) // 2
val_cases = test_val_cases[:val_size]
test_cases = test_val_cases[val_size:]

print(f"Total cases: {len(case_folders)}")
print(f"Training cases: {len(train_cases)}")
print(f"Validation cases: {len(val_cases)}")
print(f"Testing cases: {len(test_cases)}")

# Create test.txt file with case names
with open(os.path.join(list_dir, 'test_vol.txt'), 'w') as f:
    for case in test_cases:
        f.write(f"{case}\n")

# Create all.lst file (not strictly needed for direct reading approach but kept for compatibility)
with open(os.path.join(list_dir, 'all.lst'), 'w') as f:
    for case in case_folders:
        f.write(f"{case}.nii.gz\n")

# Process and create train.txt with slices
train_slices = process_cases_for_slices(train_cases, "Processing training cases")
with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.writelines(train_slices)

# Process and create val.txt with slices
val_slices = process_cases_for_slices(val_cases, "Processing validation cases")
with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
    f.writelines(val_slices)

print("List preparation completed!")