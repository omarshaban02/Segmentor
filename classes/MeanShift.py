import numpy as np
import cv2
from scipy.spatial.distance import cdist


def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * ((distance / bandwidth) ** 2))


def mean_shift(image, bandwidth=21, max_iterations=50, convergence_threshold=1e-3, sample_ratio=0.1):
    # Add suitable value of down sample ratio
    if image.shape[0] >= 1000 or image.shape[1] >= 1000:
        down_sample_ratio = 0.1

    elif image.shape[0] >= 500 or image.shape[1] >= 500:
        down_sample_ratio = 0.3

    elif image.shape[0] >= 200 or image.shape[1] >= 200:
        down_sample_ratio = 0.5

    elif image.shape[0] >= 100 or image.shape[1] >= 100:
        down_sample_ratio = 0.7

    else:
        down_sample_ratio = 1.0

    # Down sample the image
    down_sampled_image = cv2.resize(image, None, fx=down_sample_ratio, fy=down_sample_ratio)

    # Flatten the down sampled image to a list of RGB vectors
    flattened_image = down_sampled_image.reshape((-1, 3)).astype(np.float32)

    # Select a random subset of points for computing distances
    sample_size = int(sample_ratio * flattened_image.shape[0])
    sample_indices = np.random.choice(flattened_image.shape[0], size=sample_size, replace=False)
    sample_points = flattened_image[sample_indices]

    # Initialize points as their own cluster centers
    cluster_centers = sample_points.copy()

    for _ in range(max_iterations):
        # Compute distances between points and cluster centers
        distances = cdist(sample_points, cluster_centers)

        # Apply Gaussian kernel to distances
        weights = gaussian_kernel(distances, bandwidth)

        # Normalize weights
        weights /= np.sum(weights, axis=1)[:, np.newaxis]

        # Calculate weighted mean shift
        shift = np.sum(weights[:, :, np.newaxis] * sample_points[:, np.newaxis], axis=0) - cluster_centers

        # Update cluster centers
        new_cluster_centers = cluster_centers + shift

        # Check for convergence
        if np.linalg.norm(new_cluster_centers - cluster_centers) < convergence_threshold:
            break

        cluster_centers = new_cluster_centers

    # Compute distances between all points and final cluster centers
    all_distances = cdist(flattened_image, cluster_centers)

    # Assign labels based on final cluster centers
    labels = np.argmin(all_distances, axis=1)

    # Reshape labels to match down sampled image shape
    segmented_image = labels.reshape(down_sampled_image.shape[:2])

    # Up sample the segmented image to match the original image size
    segmented_image = cv2.resize(segmented_image.astype(np.uint8), image.shape[:2][::-1],
                                 interpolation=cv2.INTER_NEAREST)

    # Assign unique color to each cluster label
    unique_labels = np.unique(segmented_image)
    num_clusters = len(unique_labels)
    color_map = np.random.randint(0, 256, (num_clusters, 3))  # Generate random colors for each cluster

    # Create a colorized version of the segmented image
    colored_segmentation = np.zeros_like(image)
    for i, label in enumerate(unique_labels):
        colored_segmentation[segmented_image == label] = color_map[i]

    return colored_segmentation
