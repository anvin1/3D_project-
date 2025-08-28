#This code for checking the match between the depth and color image
#2025_08_28, it was generated for the purpose of matching due to asynchornous data collecting

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize, warp, AffineTransform
from skimage.exposure import equalize_hist  # For contrast enhancement
from skimage.registration import phase_cross_correlation  # For shift detection
from PIL import Image

# Load images (replace paths with your own)
rgb_path = r'D:\An\ZEDCAM\20250826_data2\color\frame000000.jpg'  # Your RGB image file
depth_path = r'D:\An\ZEDCAM\20250826_data2\depth\depth000235.png'  # Your depth map file

# Load RGB and convert to grayscale, then equalize
rgb_img = np.array(Image.open(rgb_path).convert('L')) / 255.0  # Normalize to [0,1]
rgb_img = equalize_hist(rgb_img)  # Enhance contrast

# Load depth map (could be grayscale or RGB)
depth_array = np.array(Image.open(depth_path)) / 255.0  # Normalize to [0,1]

# Step 1: Edge detection using Sobel filter
def detect_edges(channel):
    sobel_x = ndimage.sobel(channel, axis=0, mode='constant')
    sobel_y = ndimage.sobel(channel, axis=1, mode='constant')
    edges = np.hypot(sobel_x, sobel_y)
    return edges / np.max(edges) if np.max(edges) > 0 else edges  # Normalize

# For RGB: edges on grayscale
rgb_edges = detect_edges(rgb_img)

# For depth: handle based on dimensions
if depth_array.ndim == 2:  # Grayscale depth map
    depth_img = equalize_hist(depth_array)
    depth_edges = detect_edges(depth_img)
    depth_viz = depth_img  # For visualization
elif depth_array.ndim == 3 and depth_array.shape[2] in [3, 4]:  # RGB or RGBA
    # Equalize each channel
    for ch in range(3):  # Ignore alpha if present
        depth_array[:,:,ch] = equalize_hist(depth_array[:,:,ch])
    edges_r = detect_edges(depth_array[:,:,0])
    edges_g = detect_edges(depth_array[:,:,1])
    edges_b = detect_edges(depth_array[:,:,2])
    depth_edges = np.maximum.reduce([edges_r, edges_g, edges_b])  # Combine by max
    depth_viz = np.mean(depth_array[:,:,:3], axis=2)  # Average channels for grayscale viz
else:
    raise ValueError("Depth map has unexpected dimensions.")

# Step 2: Detect shift using phase cross-correlation on equalized grayscales
shift, error, phasediff = phase_cross_correlation(rgb_img, depth_viz, upsample_factor=10)
shift_y, shift_x = shift
print(f"Detected shift (y, x): ({shift_y:.2f}, {shift_x:.2f}) pixels")

# Step 3: Apply shift using AffineTransform
tf = AffineTransform(translation=(-shift_x, -shift_y))  # Translation to align depth to RGB
depth_edges_shifted = warp(depth_edges, tf, mode='constant', preserve_range=True)

# Apply to viz as well
depth_viz_shifted = warp(depth_viz, tf, mode='constant', preserve_range=True)

# Step 4: Resize if needed (unlikely after warp, but check)
if rgb_edges.shape != depth_edges_shifted.shape:
    depth_edges_shifted = resize(depth_edges_shifted, rgb_edges.shape, anti_aliasing=True)
if rgb_img.shape != depth_viz_shifted.shape:
    depth_viz_shifted = resize(depth_viz_shifted, rgb_img.shape, anti_aliasing=True)

# Step 5: Compute SSIM on aligned edges
score, _ = ssim(rgb_edges, depth_edges_shifted, full=True, data_range=1.0)
print(f"SSIM Score after alignment: {score:.4f}")  # >0.8 indicates good match

# Optional: Visualize with shifted depth
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(rgb_img, cmap='gray')
axs[0].set_title('RGB Grayscale (Equalized)')
axs[1].imshow(depth_viz, cmap='gray')
axs[1].set_title('Depth Pseudo-Grayscale (Original)')
axs[2].imshow(depth_viz_shifted, cmap='gray')
axs[2].set_title('Depth Pseudo-Grayscale (Shifted)')
axs[3].imshow(np.abs(rgb_edges - depth_edges_shifted), cmap='hot')
axs[3].set_title('Edge Difference (Aligned)')
plt.show()

# Interpretation
if score > 0.8:
    print("The depth map likely belongs to the RGB image (high structural match).")
elif score > 0.5:
    print("Possible match, but check for offsets or noise.")
else:
    print("Unlikely match; structures don't align well.")
print(f"If the shift is small (e.g., <5 pixels), it's likely calibration noise; larger may indicate misalignment.")
