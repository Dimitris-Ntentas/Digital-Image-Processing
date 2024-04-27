import numpy as np


# Global histogram equalization function
def get_equalization_transform_of_img(img_array: np.ndarray) -> np.ndarray:
    # Number of intensity levels in the image
    pixel_value = 256

    # Histogram
    histogram = np.zeros(pixel_value)

    # Histogram values
    for value in img_array.flatten():
        histogram[value] += 1

    # Normalization
    num_pixels = img_array.size
    prob_density = histogram / num_pixels

    # Cumulative distribution
    uk = np.zeros(pixel_value)
    cumulative_sum = 0
    for i in range(pixel_value):
        cumulative_sum += prob_density[i]
        uk[i] = cumulative_sum

    u0 = uk.min()

    # Equalization transform
    equalization_transform = np.round(((uk - u0) / (1 - u0)) * (pixel_value - 1)).astype(np.uint8)

    return equalization_transform


# Equalization transformation of the given image
def perform_global_hist_equalization(img_array: np.ndarray) -> np.ndarray:
    transformation = get_equalization_transform_of_img(img_array)
    # Apply the transformation to the image
    equalized_img = transformation[img_array]
    return equalized_img


# Calculate the equalization transformations for each contextual region
def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int, region_len_w: int) -> dict[tuple, np.ndarray]:
    h, w = img_array.shape
    region_to_eq_transform = {}
    for i in range(0, h, region_len_h):
        for j in range(0, w, region_len_w):
            # Divide the image into regions
            region = img_array[i:i + region_len_h, j:j + region_len_w]
            # Get the equalization transformation of the region
            transformation = get_equalization_transform_of_img(region)
            # Store the transformation in the dictionary
            region_to_eq_transform[(i, j)] = transformation
    return region_to_eq_transform


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int, region_len_w: int) -> np.ndarray:
    h, w = img_array.shape
    equalized_img = np.zeros_like(img_array)

    # Calculate transformations for each region
    region_transforms = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    for i in range(h):  # 360
        for j in range(w):  # 480

            # position of the region
            region_i = (i // region_len_h) * region_len_h
            region_j = (j // region_len_w) * region_len_w

            # if the pixel is on the border of the image
            if i < region_len_h // 2 or i >= h - region_len_h // 2 or j < region_len_w // 2 or j >= w - region_len_w // 2:
                equalized_img[i, j] = region_transforms[region_i, region_j][img_array[i, j]]
            else:
                # position in contextual region
                local_i = i - region_i
                local_j = j - region_j

                if local_i < region_len_h // 2 and local_j < region_len_w // 2:
                    # top-left
                    new_region = (
                        (region_i - region_len_h, region_j - region_len_w), (region_i - region_len_h, region_j),
                        (region_i, region_j - region_len_w), (region_i, region_j)
                    )
                elif local_i < region_len_h // 2 and local_j >= region_len_w // 2:
                    # top-right
                    new_region = (
                        (region_i - region_len_h, region_j), (region_i - region_len_h, region_j + region_len_w),
                        (region_i, region_j), (region_i, region_j + region_len_w)
                    )
                elif local_i >= region_len_h // 2 and local_j < region_len_w // 2:
                    # bottom-left
                    new_region = (
                        (region_i, region_j - region_len_w), (region_i, region_j),
                        (region_i + region_len_h, region_j - region_len_w), (region_i + region_len_h, region_j)
                    )
                elif local_i >= region_len_h // 2 and local_j >= region_len_w // 2:
                    # bottom-right
                    new_region = (
                        (region_i, region_j), (region_i, region_j + region_len_w), (region_i + region_len_h, region_j),
                        (region_i + region_len_h, region_j + region_len_w)
                    )

                # extract the regions from the tuple
                top_left = new_region[0]
                top_right = new_region[1]
                bottom_left = new_region[2]
                bottom_right = new_region[3]

                # get the transformations for each region
                tl = region_transforms[top_left][img_array[i, j]]
                tr = region_transforms[top_right][img_array[i, j]]
                bl = region_transforms[bottom_left][img_array[i, j]]
                br = region_transforms[bottom_right][img_array[i, j]]

                # bilinear interpolation
                w_plus = top_right[1] + region_len_w // 2
                w_minus = top_left[1] + region_len_w // 2
                h_plus = bottom_left[0] + region_len_h // 2
                h_minus = top_left[0] + region_len_h // 2

                a = (j - w_minus) / (w_plus - w_minus)
                b = (i - h_minus) / (h_plus - h_minus)

                equalized_img[i, j] = (1 - a) * (1 - b) * tl + a * (1 - b) * tr + (1 - a) * b * bl + a * b * br

    return equalized_img
