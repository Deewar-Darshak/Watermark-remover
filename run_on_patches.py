# -*- coding: utf-8 -*-
"""
Processes images by applying a Generator model patch by patch with overlap,
reconstructing the full image with averaging in overlapping regions.
Includes extensive progress bars using tqdm.
"""

import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from tqdm import tqdm  # Import tqdm

# --- Configuration (Assuming these are defined in config.py) ---
# Fallback defaults if config.py is not available or lacks definitions
try:
    from config import DEVICE, CHECKPOINT_GEN, PATCH_KERNEL_SIZE, PATCH_STRIDE
except ImportError:
    print("Warning: config.py not found or missing definitions. Using default values.")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_GEN = "gen_epoch_2.pth.tar" # Default checkpoint name
    PATCH_KERNEL_SIZE = 256
    PATCH_STRIDE = 128 # Default stride (50% overlap for 256 kernel)

# --- Model Definition (Assuming this is defined in generator_model.py) ---
# Dummy Generator if generator_model.py is not available
try:
    from generator_model import Generator
except ImportError:
    print("Warning: generator_model.py not found. Using dummy Generator.")
    import torch.nn as nn
    class Generator(nn.Module):
        def __init__(self, in_channels=3, features=64):
            super().__init__()
            self.dummy = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            print("WARNING: Using a dummy Generator model. Output will likely be incorrect.")
        def forward(self, x):
            return self.dummy(x) # Simple pass-through for testing structure

# --- Constants ---
DEFAULT_INPUT_DIR = "dataset-smol/test_images"
DEFAULT_OUTPUT_DIR = "dataset-smol/output_results_tqdm"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --- Transforms ---
# Test transforms (no data augmentation)
test_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2()
])


# --- Functions ---

def load_model(checkpoint_path: str, device: str) -> Generator:
    """
    Loads the Generator model from a specified checkpoint file.

    Args:
        checkpoint_path (str): Path to the generator checkpoint file (.pth.tar).
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        Generator: The loaded and evaluation-ready generator model.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If the checkpoint file is missing the 'state_dict'.
    """
    print(f"Loading model from: {checkpoint_path} onto device: {device}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = Generator(in_channels=3, features=64).to(device)
    try:
        # Load checkpoint allowing for pickled objects if necessary (weights_only=False)
        # Consider weights_only=True for security if you trust the source implicitly.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        raise

    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint file {checkpoint_path} is missing the 'state_dict'.")

    # Handle potential 'module.' prefix if saved with DataParallel
    new_state_dict = OrderedDict()
    has_module_prefix = any(k.startswith("module.") for k in checkpoint["state_dict"])

    for k, v in checkpoint["state_dict"].items():
        name = k.replace("module.", "") if has_module_prefix else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("This might be due to model architecture mismatch or a corrupted checkpoint.")
        raise

    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
    return model


def calculate_padding(img_h: int, img_w: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """
    Calculates the necessary padding (height, width) for an image so that
    patching with the given kernel size and stride covers the entire image.

    Args:
        img_h (int): Original image height.
        img_w (int): Original image width.
        kernel_size (int): The size of the square patches (height and width).
        stride (int): The step size between patches.

    Returns:
        tuple[int, int]: The required padding for height (pad_h) and width (pad_w).
                         Padding is added to the bottom and right.
    """
    # Calculate padding required to make dimensions perfectly divisible by stride after
    # the first kernel window is placed.
    # If the image dimension is smaller than the kernel, pad to kernel size.
    pad_h = kernel_size - img_h if img_h < kernel_size else (stride - (img_h - kernel_size) % stride) % stride
    pad_w = kernel_size - img_w if img_w < kernel_size else (stride - (img_w - kernel_size) % stride) % stride

    # Ensure minimal padding if already perfectly divisible
    # Example: H=288, kernel=256, stride=32. (288-256)%32 = 32%32 = 0. (32-0)%32 = 0. Correct.
    # Example: H=256, kernel=256, stride=32. (256-256)%32 = 0%32 = 0. (32-0)%32 = 0. Correct.
    # Example: H=300, kernel=256, stride=32. (300-256)%32 = 44%32 = 12. (32-12)%32 = 20. Correct. Need 20 padding.

    return pad_h, pad_w


def process_image(model: Generator, image_path: str, output_dir: str,
                  kernel_size: int, stride: int, device: str, M_name: str):
    """
    Processes a single image by dividing it into patches, running the model
    on each patch, and reconstructing the output image with averaging.
    Includes tqdm progress bars for patch processing and reconstruction.

    Args:
        model (Generator): The loaded generator model.
        image_path (str): Path to the input image file.
        output_dir (str): Directory to save the processed image.
        kernel_size (int): The size of the square patches.
        stride (int): The step size between patches.
        device (str): The device to perform computations on ('cuda' or 'cpu').
    """
    image_filename = M_name.replace(".pth.tar","") + "_" + os.path.basename(image_path)
    print(f"\nProcessing: {image_filename}")

    # --- 1. Load and Transform Image ---
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        H, W, _ = image_np.shape
        print(f"  Original dimensions: {W}x{H}")
    except FileNotFoundError:
        print(f"  Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"  Error loading image {image_path}: {e}")
        return

    transformed = test_transform(image=image_np)
    input_tensor = transformed['image'].to(device) # Shape: (C, H, W)
    C = input_tensor.shape[0]

    # --- 2. Calculate Padding ---
    pad_h, pad_w = calculate_padding(H, W, kernel_size, stride)
    print(f"  Calculated padding (H, W): ({pad_h}, {pad_w})")

    # --- 3. Pad Image ---
    # Padding format: (pad_left, pad_right, pad_top, pad_bottom)
    padded_tensor = F.pad(input_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    _, H_pad, W_pad = padded_tensor.shape
    print(f"  Padded dimensions: {W_pad}x{H_pad}")

    # --- 4. Extract Patches ---
    # unfold: (dimension, size, step)
    try:
        patches = padded_tensor.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        # patches shape: (C, num_patches_h, num_patches_w, kernel_size, kernel_size)
        num_patches_h = patches.shape[1]
        num_patches_w = patches.shape[2]
        num_patches_total = num_patches_h * num_patches_w
        print(f"  Extracted {num_patches_total} patches ({num_patches_h} H x {num_patches_w} W)")

        # Reshape for processing: (num_patches_total, C, kernel_size, kernel_size)
        patches = patches.contiguous().view(C, -1, kernel_size, kernel_size)
        patches = patches.permute(1, 0, 2, 3).contiguous()
    except RuntimeError as e:
         print(f"  Error unfolding patches (check kernel/stride vs image size): {e}")
         print(f"  Image H/W: {H}/{W}, Padded H/W: {H_pad}/{W_pad}, Kernel: {kernel_size}, Stride: {stride}")
         return # Skip this image

    # --- 5. Process Patches with Model ---
    output_patches = []
    # Use tqdm for patch processing progress
    pbar_patches = tqdm(patches, total=num_patches_total,
                        desc=f"  Inferring patches for {image_filename}",
                        unit="patch", leave=True)
    with torch.no_grad():
        for patch in pbar_patches:
            # Add batch dimension, run model, remove batch dimension
            output_patch = model(patch.unsqueeze(0)).squeeze(0)
            output_patches.append(output_patch.cpu()) # Move to CPU to save GPU memory

    # Stack output patches: (num_patches_total, C, kernel_size, kernel_size)
    output_patches = torch.stack(output_patches).to(device) # Move back to device for reconstruction

    # --- 6. Reconstruct Output Image ---
    # Initialize output tensor and count tensor (for averaging overlaps)
    output_tensor = torch.zeros((C, H_pad, W_pad), device=device)
    count_tensor = torch.zeros((C, H_pad, W_pad), device=device)

    # Use tqdm for reconstruction progress (iterating through patch grid)
    patch_idx = 0
    pbar_reconstruct = tqdm(total=num_patches_total,
                            desc=f"  Reconstructing {image_filename}",
                            unit="patch", leave=True)

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size

            # Add the processed patch to the output tensor
            output_tensor[:, h_start:h_end, w_start:w_end] += output_patches[patch_idx]
            # Increment the count for the corresponding region
            count_tensor[:, h_start:h_end, w_start:w_end] += 1

            patch_idx += 1
            pbar_reconstruct.update(1)

    pbar_reconstruct.close()

    # --- 7. Average Overlapping Regions and Crop ---
    # Avoid division by zero if somehow a count is zero (shouldn't happen with correct padding)
    output_averaged = (output_tensor / count_tensor.clamp(min=1e-6))

    # Crop back to original dimensions
    output_cropped = output_averaged[:, :H, :W]
    print(f"  Final output dimensions: {output_cropped.shape[2]}x{output_cropped.shape[1]}")

    # --- 8. Convert to PIL Image and Save ---
    # Permute from (C, H, W) to (H, W, C) for numpy/PIL
    output_numpy = output_cropped.permute(1, 2, 0).cpu().numpy()

    # Denormalize: [-1, 1] -> [0, 1] -> [0, 255]
    output_numpy = (output_numpy * 0.5 + 0.5) * 255.0

    # Clip values to be safe and convert to uint8
    output_numpy = output_numpy.clip(0, 255).astype(np.uint8)

    output_image = Image.fromarray(output_numpy)

    # Save the result
    # image_filename
    output_path = os.path.join(output_dir, image_filename)
    try:
        output_image.save(output_path)
        print(f"  Saved result to: {output_path}")
    except Exception as e:
        print(f"  Error saving image {output_path}: {e}")


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting Patch-Based Image Processing ---")
    print(f"Using device: {DEVICE}")
    print(f"Using patch kernel size: {PATCH_KERNEL_SIZE}")
    print(f"Using patch stride: {64}")
    M = "gen_epoch_9.pth.tar"

    # --- Initialize Model ---
    try:
        model = load_model(M, DEVICE)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"Fatal Error: Could not load model. Exiting. Reason: {e}")
        exit(1) # Exit if model loading fails

    # --- Setup Directories ---
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found. Please create it or modify DEFAULT_INPUT_DIR.")
        exit(1)

    os.makedirs(output_dir, exist_ok=True) # Create output dir if it doesn't exist

    # --- Find Image Files ---
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files.append(filename)

    if not image_files:
        print(f"No supported image files {SUPPORTED_EXTENSIONS} found in '{input_dir}'.")
        exit(0)

    print(f"Found {len(image_files)} images to process.")

    # --- Process All Images ---
    # Use tqdm for the main loop over images
    main_pbar = tqdm(image_files, total=len(image_files),
                     desc="Overall Image Processing", unit="image", leave=True)

    for filename in main_pbar:
        img_path = os.path.join(input_dir, filename)
        # Update description for the main progress bar
        main_pbar.set_description(f"Processing {filename[:20]}...") # Show truncated filename

        try:
            process_image(
                model=model,
                image_path=img_path,
                output_dir=output_dir,
                kernel_size=PATCH_KERNEL_SIZE,
                stride=64,
                device=DEVICE,
                M_name = M
            )
        except Exception as e:
            # Catch any unexpected errors during processing of a single image
            print(f"\n--- !!! Critical Error processing {filename} !!! ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            print(f"--- Skipping {filename} and continuing... ---")
            # Optionally add to a list of failed files here
            continue # Move to the next image

    print("\n--- Processing Complete ---")