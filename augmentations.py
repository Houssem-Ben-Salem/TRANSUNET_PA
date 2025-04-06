import random
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import torch
import elasticdeform.torch as etorch

class EnhancedRandomGenerator(object):
    def __init__(self, output_size, elastic_deform_prob=0.15, gamma_prob=0.3, noise_prob=0.2, contrast_prob=0.2):
        """
        Enhanced random generator for PA segmentation with specialized augmentations
        
        Args:
            output_size: Target output size (h, w)
            elastic_deform_prob: Probability of applying elastic deformation
            gamma_prob: Probability of applying random gamma correction
            noise_prob: Probability of adding random noise
            contrast_prob: Probability of enhancing local contrast
        """
        self.output_size = output_size
        self.elastic_deform_prob = elastic_deform_prob
        self.gamma_prob = gamma_prob
        self.noise_prob = noise_prob
        self.contrast_prob = contrast_prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Basic augmentations (rotation/flip)
        if random.random() > 0.5:
            image, label = self._random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self._random_rotate(image, label)
            
        # Elastic deformation (specifically good for vascular structures)
        if random.random() < self.elastic_deform_prob:
            # Convert to torch tensors for elasticdeform
            image_t = torch.from_numpy(image.astype(np.float32))
            label_t = torch.from_numpy(label.astype(np.float32))
            
            # Stack for joint deformation
            stacked = torch.stack([image_t, label_t], dim=0)
            
            # Apply elastic deformation
            sigma = random.uniform(10, 15)  # Controls smoothness
            alpha = random.uniform(100, 120)  # Controls intensity
            
            # Apply deformation using the elasticdeform library
            try:
                deformed = etorch.deform_random_grid(
                    stacked, sigma=sigma, points=3, order=[3, 0]
                )
                image = deformed[0].numpy()
                label = deformed[1].numpy().astype(np.int32)
            except:
                # Fallback in case of error with elastic deform
                pass
        
        # Random gamma correction (intensity variation)
        if random.random() < self.gamma_prob:
            gamma = random.uniform(0.8, 1.2)
            image_min = image.min()
            image = np.power((image - image_min + 1e-8) / (image.max() - image_min + 1e-8), gamma) 
            
        # Add random noise
        if random.random() < self.noise_prob:
            noise_level = random.uniform(0, 0.02)
            noise = np.random.normal(0, noise_level, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
            
        # Local contrast enhancement (helps with small bright structures like vessels)
        if random.random() < self.contrast_prob:
            # Create a blurred version
            blurred = ndimage.gaussian_filter(image, sigma=1.5)
            
            # Calculate local contrast
            local_contrast = image - blurred
            
            # Enhance the local contrast
            enhance_factor = random.uniform(1.2, 1.5)
            enhanced = image + local_contrast * enhance_factor
            
            # Clip and normalize
            enhanced = np.clip(enhanced, 0, 1)
            image = enhanced
            
        # Resize to target size if needed
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample
    
    def _random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def _random_rotate(self, image, label):
        # Wider rotation range for better augmentation
        angle = np.random.randint(-30, 30)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label