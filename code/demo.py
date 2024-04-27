from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from main import perform_global_hist_equalization, perform_adaptive_hist_equalization

# Load an image
img_path = 'input_img.png'
img = Image.open(img_path)

# Convert the image to grayscale
bw_img = img.convert('L')
img_array = np.array(bw_img)
original_img = Image.fromarray(img_array)
original_img.show(title='Original Image')

# Plot histogram of the original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.hist(img_array.ravel(), bins=256, color='gray', alpha=0.75)
plt.title('Original Image Histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')

# Apply global histogram equalization
global_eq_img_array = perform_global_hist_equalization(img_array)
global_eq_img = Image.fromarray(global_eq_img_array)
global_eq_img.show(title='Global Histogram Equalization')

# Plot histogram of the globally equalized image
plt.subplot(1, 3, 2)
plt.hist(global_eq_img_array.ravel(), bins=256, color='blue', alpha=0.75)
plt.title('Globally Equalized Histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')

# Define region sizes for adaptive histogram equalization (AHE)
region_len_h = 48
region_len_w = 64

# Apply adaptive histogram equalization
adaptive_eq_img_array = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)
adaptive_eq_img = Image.fromarray(adaptive_eq_img_array)
adaptive_eq_img.show(title='Adaptive Histogram Equalization')

# Plot histogram of the adaptively equalized image
plt.subplot(1, 3, 3)
plt.hist(adaptive_eq_img_array.ravel(), bins=256, color='green', alpha=0.75)
plt.title('Adaptively Equalized Histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
