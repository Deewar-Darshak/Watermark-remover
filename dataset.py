import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class WatermarkPatchDataset(Dataset):
    def __init__(self, mark_dir, nomark_dir, kernel_size=256, stride=32,
                 transform_input=None, transform_target=None):
        self.mark_dir = mark_dir
        self.nomark_dir = nomark_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.transform_input = transform_input
        self.transform_target = transform_target

        self.mark_files = sorted([f for f in os.listdir(mark_dir) if f.endswith('_c.jpg')])

        self.paired_patches = []
        self._prepare_all_patches()

    def _prepare_all_patches(self):
        for mark_file in self.mark_files:
            base_name = mark_file.replace('_c.jpg', '')
            mark_path = os.path.join(self.mark_dir, mark_file)
            nomark_path = os.path.join(self.nomark_dir, f'{base_name}_r.jpg')

            mark_img = Image.open(mark_path).convert('RGB')
            nomark_img = Image.open(nomark_path).convert('RGB')
            nomark_img = nomark_img.resize(mark_img.size, Image.LANCZOS)

            # Convert to numpy
            mark_np = np.array(mark_img)
            nomark_np = np.array(nomark_img)

            # Apply transformations
            mark_tensor = self.transform_input(image=mark_np)['image'] if self.transform_input else torch.from_numpy(mark_np).permute(2, 0, 1).float() / 255.
            nomark_tensor = self.transform_target(image=nomark_np)['image'] if self.transform_target else torch.from_numpy(nomark_np).permute(2, 0, 1).float() / 255.

            # Extract patches
            _, H, W = mark_tensor.shape
            mark_patches = mark_tensor.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride)
            nomark_patches = nomark_tensor.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride)

            mark_patches = mark_patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, self.kernel_size, self.kernel_size)
            nomark_patches = nomark_patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, self.kernel_size, self.kernel_size)

            self.paired_patches.extend(list(zip(mark_patches, nomark_patches)))

    def __len__(self):
        return len(self.paired_patches)

    def __getitem__(self, idx):
        return self.paired_patches[idx]  # (input_patch, target_patch)
