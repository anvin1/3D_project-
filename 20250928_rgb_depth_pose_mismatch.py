#  this code for fixing data , mismatch depth and rgb pose
# input is rgb, depth and pose folder
$ tune the shifr pixel for higher accuracy

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.transform import resize, warp, AffineTransform
from skimage.exposure import equalize_hist
from skimage.registration import phase_cross_correlation
from PIL import Image
import os
import re  # For extracting numeric indices from filenames
import shutil  # For copying files

# Folders (update these paths)
color_dir = r'D:\An\ZEDCAM\20250826_data2\color - Copy'  # Folder with color images (e.g., frame000.jpg)
depth_dir = r'D:\An\ZEDCAM\20250826_data2\depth'  # Folder with depth images (e.g., frame000.png)
output_dir = 'D:/An/ZEDCAM/matched_outputs'  # Where to save outputs
os.makedirs(output_dir, exist_ok=True)

# New folders for matched pairs
matched_color_dir = os.path.join(output_dir, 'matched_color')
matched_depth_dir = os.path.join(output_dir, 'matched_depth')
os.makedirs(matched_color_dir, exist_ok=True)
os.makedirs(matched_depth_dir, exist_ok=True)

# Path to original traj.txt (assumed in parent of depth_dir; adjust if needed)
traj_path = os.path.join(os.path.dirname(depth_dir), 'traj.txt')
new_traj_path = os.path.join(output_dir, 'matched_traj.txt')

# Parameters
max_offset = 20  # Check up to +20 above expected
max_shift_pixels = 15  # Threshold for shift magnitude (pixels); below this = good match
overlay_alpha = 0.5  # Transparency for depth overlay in combined image

# List and sort files, extract indices
def get_files_with_indices(directory, extension='.jpg'):
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    # Sort by numeric index (e.g., frame000.jpg -> 0)
    files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(1)) if re.search(r'(\d+)', f) else 0)
    indices = [int(re.search(r'(\d+)', f).group(1)) for f in files if re.search(r'(\d+)', f)]
    return list(zip(indices, [os.path.join(directory, f) for f in files]))

color_files = get_files_with_indices(color_dir, '.jpg')  # Assuming .jpg for color
depth_files = get_files_with_indices(depth_dir, '.png')  # Assuming .png for depth

# Process depth array (equalize and get viz; edges removed since no SSIM)
def process_depth(depth_path):
    depth_array = np.array(Image.open(depth_path)) / 255.0
    if depth_array.ndim == 2:
        depth_viz = equalize_hist(depth_array)
    elif depth_array.ndim == 3 and depth_array.shape[2] in [3, 4]:
        for ch in range(3):
            depth_array[:,:,ch] = equalize_hist(depth_array[:,:,ch])
        depth_viz = np.mean(depth_array[:,:,:3], axis=2)
    else:
        raise ValueError("Depth map has unexpected dimensions.")
    return depth_viz

# Main loop over color images
matched_pairs = []
expected_offset = 0  # Start with no offset; update based on matches
new_frame_idx = 0  # For renumbering matched pairs
for color_idx, color_path in color_files:
    print(f"Processing color frame {color_idx}...")
    rgb_img = np.array(Image.open(color_path).convert('L')) / 255.0
    rgb_img = equalize_hist(rgb_img)
    
    best_shift_mag = float('inf')
    best_depth_idx = None
    best_depth_path = None
    best_shifted_viz = None
    
    # Forward search from expected onwards, up to +max_offset
    expected_depth = color_idx + expected_offset
    start_depth_idx = max(color_idx, expected_depth)
    end_depth_idx = min(start_depth_idx + max_offset, max(d[0] for d in depth_files))  # Cap at available
    
    for depth_idx, depth_path in [d for d in depth_files if start_depth_idx <= d[0] <= end_depth_idx]:
        depth_viz = process_depth(depth_path)
        
        # Detect shift
        shift, _, _ = phase_cross_correlation(rgb_img, depth_viz, upsample_factor=10)
        shift_y, shift_x = shift
        shift_mag = np.sqrt(shift_y**2 + shift_x**2)
        
        # Prioritize smallest shift
        if shift_mag < best_shift_mag:
            best_shift_mag = shift_mag
            best_depth_idx = depth_idx
            best_depth_path = depth_path
            tf = AffineTransform(translation=(-shift_x, -shift_y))
            best_shifted_viz = warp(depth_viz, tf, mode='constant', preserve_range=True)
            if best_shift_mag < max_shift_pixels:
                print(f"Early match found for color {color_idx} with depth {best_depth_idx} (shift mag: {best_shift_mag:.2f})")
                break  # Stop if good enough
    
    if best_shift_mag < max_shift_pixels:
        matched_pairs.append((color_idx, best_depth_idx))
        print(f"Matched color {color_idx} to depth {best_depth_idx} with shift mag {best_shift_mag:.2f}")
        
        # Update expected_offset for next frame
        expected_offset = best_depth_idx - color_idx
        
        # Load originals for output
        orig_color = np.array(Image.open(color_path))
        orig_depth = np.array(Image.open(best_depth_path))
        
        # Combined: Overlay (colorize depth and blend)
        if orig_depth.ndim == 2:
            orig_depth_colored = plt.cm.viridis((orig_depth / np.max(orig_depth)))[:,:,:3]  # Colorize if grayscale, normalize
        else:
            orig_depth_colored = orig_depth / 255.0  # Already color
        # Resize if needed
        if orig_color.shape[:2] != orig_depth_colored.shape[:2]:
            orig_depth_colored = resize(orig_depth_colored, orig_color.shape[:2], anti_aliasing=True)
        combined_overlay = (orig_color / 255.0 * (1 - overlay_alpha) + orig_depth_colored * overlay_alpha) * 255
        combined_overlay = combined_overlay.astype(np.uint8)
        
        # Side-by-side combined: Ensure both are 3D uint8
        orig_depth_for_side = orig_depth.copy()
        if orig_depth_for_side.ndim == 2 or orig_depth_for_side.dtype == np.uint16:
            # Normalize to uint8 grayscale
            orig_depth_for_side = ((orig_depth_for_side - orig_depth_for_side.min()) / (orig_depth_for_side.max() - orig_depth_for_side.min() + 1e-8) * 255).astype(np.uint8)
        if orig_depth_for_side.ndim == 2:
            orig_depth_for_side = np.stack([orig_depth_for_side] * 3, axis=-1)  # To RGB
        # Resize if needed
        if orig_color.shape[:2] != orig_depth_for_side.shape[:2]:
            orig_depth_for_side = resize(orig_depth_for_side / 255.0, orig_color.shape[:2], anti_aliasing=True) * 255
            orig_depth_for_side = orig_depth_for_side.astype(np.uint8)
        side_by_side = np.hstack((orig_color, orig_depth_for_side))
        
        # Save combined side-by-side (only this as per request; comment out others if not needed)
        Image.fromarray(side_by_side).save(os.path.join(output_dir, f'combined_sidebyside_{color_idx:03d}_depth_{best_depth_idx:03d}.jpg'))
        
        # Copy to new folders with renumbered names
        new_color_path = os.path.join(matched_color_dir, f'frame{new_frame_idx:06d}.jpg')
        new_depth_path = os.path.join(matched_depth_dir, f'depth{new_frame_idx:06d}.png')
        shutil.copy(color_path, new_color_path)
        Image.fromarray(orig_depth).save(new_depth_path)
        
        new_frame_idx += 1

# After all matches, recalculate traj by selecting absolute poses at matched depths
if matched_pairs:
    # Load original traj
    traj_data = np.loadtxt(traj_path)
    traj_matrices = traj_data.reshape(-1, 4, 4)
    
    matched_depths = [d for c, d in matched_pairs]
    new_traj_flat = [traj_matrices[d].flatten() for d in matched_depths]
    
    np.savetxt(new_traj_path, new_traj_flat)
    print(f"Selected absolute traj saved to {new_traj_path}")

print("Processing complete. Check outputs in:", output_dir)
print("Matched pairs:", matched_pairs)
