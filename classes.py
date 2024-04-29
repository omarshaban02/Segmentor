import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import random
from scipy.spatial.distance import cdist


class WorkerSignals(QObject):
    get_segmented_image = pyqtSignal(np.ndarray)


class RegionGrowingThread(QThread):
    def __init__(self, input_image, seeds, threshold):
        super(RegionGrowingThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image
        self.threshold = threshold
        self.seeds = seeds

    def run(self):
        segmented_img = region_growing(self.input_image, seeds=self.seeds, threshold=self.threshold)
        self.signals.get_segmented_image.emit(segmented_img)


class MeanShiftThread(QThread):
    def __init__(self, input_image):
        super(MeanShiftThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image

    def run(self):
        segmented_img = mean_shift(self.input_image)
        self.signals.get_segmented_image.emit(segmented_img)


# ######################################## Region Growing Algorithm start ######################################
def region_growing(img, seeds=None, threshold=10):
    if seeds is None:
        seeds = find_seeds(img)

    segmented = np.zeros_like(img)
    visited = np.zeros_like(img)
    label_colors = {}

    label = 1  # Start labeling from 1
    for seed in seeds:
        seed_row, seed_col = seed
        stack = [(seed_row, seed_col)]
        while stack:
            pixel = stack.pop()
            row, col = pixel
            if visited[row, col]:
                continue
            segmented[row, col] = label
            visited[row, col] = True
            if label not in label_colors:
                label_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            neighbors = get_neighbors(pixel, img, threshold)
            for neighbor in neighbors:
                n_row, n_col = neighbor
                if not visited[n_row, n_col] and np.abs(int(img[n_row, n_col]) - int(img[row, col])) < threshold:
                    stack.append((n_row, n_col))
        label += 1  # Increment label for the next region

    # Convert labeled image to RGB
    segmented_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(segmented.shape[0]):
        for j in range(segmented.shape[1]):
            if segmented[i, j] != 0:
                segmented_rgb[i, j] = label_colors[segmented[i, j]]

    return segmented_rgb


def find_seeds(img):
    # If seeds are not provided, find the three local maxima in the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    sorted_idx = np.argsort(hist)
    seeds_intensities = [sorted_idx[-1], sorted_idx[-2], sorted_idx[-3]]  # Three largest peaks in histogram

    seed_locations = []
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            if len(seeds_intensities) == 0:
                return seed_locations
            if img[row, col] in seeds_intensities:
                seed_locations.append((row, col))
                seeds_intensities.remove(img[row, col])

    return seed_locations


def get_neighbors(pixel, img, threshold):
    neighbors = []
    rows, cols = img.shape
    row, col = pixel

    # Add orthogonal neighbors
    if row > 0:
        neighbors.append((row - 1, col))
    if row < rows - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < cols - 1:
        neighbors.append((row, col + 1))

    # Add diagonal neighbors
    if row > 0 and col > 0:
        neighbors.append((row - 1, col - 1))
    if row > 0 and col < cols - 1:
        neighbors.append((row - 1, col + 1))
    if row < rows - 1 and col > 0:
        neighbors.append((row + 1, col - 1))
    if row < rows - 1 and col < cols - 1:
        neighbors.append((row + 1, col + 1))

    valid_neighbors = []
    for neighbor in neighbors:
        neigh_row, neigh_col = neighbor
        if abs(img[neigh_row, neigh_col] - img[row, col]) <= threshold:
            valid_neighbors.append(neighbor)

    return valid_neighbors


# ######################################## Region Growing Algorithm end ######################################


# ######################################## Mean Shift Algorithm start ######################################
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

    # Reshape labels to match downsampled image shape
    segmented_image = labels.reshape(down_sampled_image.shape[:2])

    # Upsample the segmented image to match the original image size
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


# ######################################## Mean Shift Algorithm end ######################################


########################################## Thresholding ##################################################
class Thresholding:
    def __init__(self, input_image):
        self.img = input_image.copy()

    def global_threshold(self, threshold_value=127, max_value=255):
        """
        Apply global thresholding to a grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            threshold_value (int): Threshold value.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Thresholded image.
        """
        image = self.img
        thresholded = np.where(image > threshold_value, max_value, 0).astype(np.uint8)
        return thresholded

    def local_threshold(self, blockSize=11, C=2, max_value=255):
        """
        Apply local thresholding to a grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            blockSize (int): Size of the local neighborhood for computing the threshold value.
            C (int): Constant subtracted from the mean or weighted mean.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Thresholded image.
        """
        image = self.img
        thresholded = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Define the region of interest
                roi = image[max(0, i - blockSize // 2): min(image.shape[0], i + blockSize // 2),
                      max(0, j - blockSize // 2): min(image.shape[1], j + blockSize // 2)]
                # Compute the threshold value for the region
                threshold_value = np.mean(roi) - C
                # Apply thresholding
                thresholded[i, j] = max_value if image[i, j] > threshold_value else 0
        return thresholded

    def _compute_otsu_criteria(self, im, th):
        # create the thresholded image
        thresholded_im = np.zeros(im.shape)
        thresholded_im[im >= th] = 1

        # compute weights
        nb_pixels = im.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = im[thresholded_im == 1]
        val_pixels0 = im[thresholded_im == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

        return weight0 * var0 + weight1 * var1

    def otsuThresholding(self):
        img = self.img
        threshold_range = range(np.max(img) + 1)
        criterias = np.array([self._compute_otsu_criteria(img, th) for th in threshold_range])

        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criterias)]

        binary = img
        binary[binary < best_threshold] = 0
        binary[binary >= best_threshold] = 255

        return binary

    def optimal_thresholding(self):
        # Convert image to grayscale
        gray_image = self.img

        # Initialize threshold with a random value (e.g., midpoint of intensity range)
        min_intensity = np.min(gray_image)
        max_intensity = np.max(gray_image)
        threshold = (min_intensity + max_intensity) // 2

        # Iterate until convergence (threshold value stabilizes)
        while True:
            # Classify pixels into foreground (class 1) and background (class 2) based on current threshold
            foreground_pixels = gray_image[gray_image > threshold]
            background_pixels = gray_image[gray_image <= threshold]

            # Calculate mean intensity values of the two classes
            mean_foreground = np.mean(foreground_pixels)
            mean_background = np.mean(background_pixels)

            # Calculate new threshold as the average of mean intensities
            new_threshold = (mean_foreground + mean_background) / 2

            # Check convergence: if new threshold is close to the old threshold, break the loop
            if np.abs(new_threshold - threshold) < 1e-3:
                break

            # Update the threshold
            threshold = new_threshold

        # Apply the final threshold to the grayscale image
        thresholded_image = (gray_image > threshold).astype(np.uint8) * 255

        return thresholded_image


# ######################################### Thresholding ends ##################################################

# ######################################### LUV mapping ##################################################

def luv_mapping(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    shape = r.shape

    b = (b / 255).flatten()
    g = (g / 255).flatten()
    r = (r / 255).flatten()

    def gamma_correction(value):
        value = np.asarray(value)
        condition = value <= 0.04045
        res = np.where(
            condition,
            value / 12.92,
            ((value + 0.055) / 1.055) ** 2.4
        )
        return res

    r = gamma_correction(r)
    g = gamma_correction(g)
    b = gamma_correction(b)
    rgb = np.array([r, g, b])

    converting_mat = [[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]]

    x, y, z = np.matmul(converting_mat, rgb)

    # Calculate the chromaticity coordinates
    u_dash = 4 * x / (x + 15 * y + 3 * z)
    v_dash = 9 * y / (x + 15 * y + 3 * z)

    un = 0.19793943
    vn = 0.46831096

    # Calculate L
    y_gt_idx = np.argwhere(y > 0.008856)
    y_le_idx = np.argwhere(y <= 0.008856)

    l = np.zeros_like(y)
    l[y_gt_idx] = (116 * y[y_gt_idx] ** (1 / 3)) - 16
    l[y_le_idx] = 903.3 * y[y_le_idx]

    # Calculate u and v
    u = 13 * l * (u_dash - un)
    v = 13 * l * (v_dash - vn)

    # Conversion to 8-bit
    l = 255 / 100 * l
    u = 225 / 354 * (u + 134)
    v = 255 / 262 * (v + 140)

    # Reshape after flattening
    l = l.reshape(shape)
    u = u.reshape(shape)
    v = v.reshape(shape)

    luv = np.array([l, u, v], np.int64).T
    luv = np.fliplr(np.rot90(luv, 3))
    return luv


# ######################################### LUV mapping ends ##################################################

# ######################################### K means ##################################################
class KMeansClustering:
    def __init__(self, K, max_iterations=100, tolerance=1e-4):
        self.K = K
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None

    def initialize_centroids(self, data):
        # Randomly choose K unique centroids from the data
        self.centroids = data[np.random.choice(data.shape[0], self.K, replace=False)]

    def calculate_distances(self, data):
        # Compute the Euclidean distances efficiently
        return np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)

    def run_kmeans_iterations(self, data):
        # Initialize centroids
        self.initialize_centroids(data)

        # Main loop for K-means iterations
        for iteration in range(self.max_iterations):
            # Calculate distances and assign clusters
            distances = self.calculate_distances(data)
            cluster_indices = np.argmin(distances, axis=1)

            # Update centroids based on mean of points in each cluster
            new_centroids = np.array(
                [data[cluster_indices == k].mean(axis=0) for k in range(self.K)]
            )

            # Check for convergence
            if np.allclose(new_centroids, self.centroids, rtol=self.tolerance):
                break  # Convergence achieved

            self.centroids = new_centroids  # Update centroids for the next iteration

        return cluster_indices, self.centroids

    def apply_to_image(self, image):
        # Flatten image to 2D array (height*width, 3) for RGB
        pixel_data = image.reshape(-1, 3).astype(float)

        # Run K-means clustering
        cluster_indices, centroids = self.run_kmeans_iterations(pixel_data)

        # Create segmented image with the centroids' colors
        segmented_image_data = centroids[cluster_indices].astype(int)

        # Reshape segmented data to original image dimensions
        segmented_image = segmented_image_data.reshape(image.shape)

        return segmented_image
