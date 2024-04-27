# Image Enhancement via Histogram Equalization

This repository contains the Python scripts and results for an image processing assignment focused on enhancing the contrast of grayscale images through histogram equalization techniques. The methodology includes both conventional and adaptive approaches to histogram equalization.

## Introduction

The aim of this report is to document the process and outcomes of applying histogram equalization techniques for enhancing the contrast in grayscale images. The methodologies evaluated include conventional and adaptive histogram equalization methods, implemented through Python scripts developed for this assignment.

## Methods

The `main.py` script implements several functions:

- `get_equalization_transforms_of_img`: Calculates the histogram equalization transformation for an input image.
- `perform_global_hist_equalization`: Applies the global histogram equalization to the entire image.
- `calculate_eq_transformations_of_regions`: Divides the image into contextual regions and computes local transformations.
- `perform_adaptive_hist_equalization`: Carries out adaptive histogram equalization based on the local regions, preserving local image details.

## Results

The provided images visualize the results of applying the histogram equalization techniques:

- Original image after grayscaling.
- Image after conventional histogram equalization.
- Image after adaptive histogram equalization.

Histograms illustrate the pixel intensity distribution for each image, demonstrating the effectiveness of the methods.

## Conclusion

The application of histogram equalization methods as shown by the images and Python scripts confirms the utility of these techniques in image processing to enhance contrast. The conventional method offers a general contrast improvement, while the adaptive method provides a more detailed enhancement respecting the image's local features.
