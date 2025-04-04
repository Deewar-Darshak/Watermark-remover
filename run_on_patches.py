import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from PIL import Image


def run_on_patches(directory, model, kernel_size=256, stride=32, device='cpu', train=False, save_results=True):
    model = model.to(device)
    if not train:
        model.eval()

        window = torch.hann_window(kernel_size, device=device)
        weight = window.unsqueeze(0)*window.unsqueeze(1)
        weight = weight.expand(3, -1, -1)

    for idx, image_file in enumerate(os.listdir(directory)):
        image = Image.open(os.path.join(directory, image_file)).convert('RGB')
        width, height = image.size

        max_size = math.ceil(max(width, height) / kernel_size) * kernel_size
        pad_height = max_size - height
        pad_width = max_size - width

        image = np.array(image)
        augment = A.Compose([
            A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_REFLECT),
            A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
        image_tensor = augment(image=image)['image'].to(device)
        img_size = image_tensor.shape[2]

        image_tensor = image_tensor.permute(1, 2, 0)

        kh, kw = kernel_size, kernel_size
        dh, dw = stride, stride

        patches = image_tensor.unfold(0, kh, dh).unfold(1, kw, dw)
        patches = patches.contiguous().view(-1, 3, kh, kw)

        batch_size = 32
        for id in tqdm(range(math.ceil(patches.shape[0] / batch_size))):
            from_idx = id * batch_size
            to_idx = min((id + 1) * batch_size, patches.shape[0])

            curr_patch = patches[from_idx:to_idx].to(device)
            if not train:
                with torch.no_grad():
                    patch = model(curr_patch)
                patch = patch * weight.unsqueeze(0)
            else:
                patch = model(curr_patch)

            patches[from_idx:to_idx] = patch.cpu()

        patches = patches.view(1, patches.shape[0], 3 * kernel_size * kernel_size).permute(0, 2, 1)
        output = F.fold(patches, output_size=(img_size, img_size), kernel_size=kernel_size, stride=dh)

        recovery_mask = F.fold(torch.ones_like(patches), output_size=(img_size, img_size),
                               kernel_size=kernel_size, stride=dh)
        output /= recovery_mask

        # Denormalize and convert to uint8
        output = output.squeeze(0)  # (3, H, W)
        output = (output * 0.5 + 0.5).clamp(0, 1)  # [0, 1]
        output_np = output.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        output_np = (output_np * 255).astype(np.uint8)

        augment_back = A.Compose([
            A.CenterCrop(height=max_size - int(pad_height), width=max_size - int(pad_width)),
            ToTensorV2(),
        ])

        cropped_np = augment_back(image=output_np)['image']

        if save_results and not train:

            Image.fromarray(cropped_np).save(f"saved/test_results_{idx}.png")

        if train:
            yield output  # Return tensor for training

    if not train:
        model.train()
    #     x = augment_back(image=output.squeeze(0).detach().cpu().permute(1, 2, 0).numpy())['image']
    #
    #     if save_results and not train:
    #         save_image(x, f"saved/test_results_{idx}.png")
    #
    #     # Optionally return the image if needed for training loss or visualization
    #     if train:
    #         yield x  # You can yield or return a list of images if batching
    #
    #
    # if not train:
    #     model.train()
