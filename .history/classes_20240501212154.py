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

    def compute_otsu_criteria(self, im, th):
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
        criterias = np.array([self.compute_otsu_criteria(img, th) for th in threshold_range])

        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criterias)]

        binary = img
        binary[binary < best_threshold] = 0
        binary[binary >= best_threshold] = 255

        return binary
    
    def local_otsu_thresholding(self, block_size = 5, sigma = 1) :
        gray_image = self.img

        
        # Apply Gaussian smoothing to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        # Get image dimensions
        height, width = gray_image.shape
        
        # Initialize the output binary image
        binary_image = np.zeros_like(gray_image)
        
        # Loop over the image with specified block size
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Define the current block within the smoothed image
                block = blurred_image[y:y+block_size, x:x+block_size]
                
                # Calculate the Otsu threshold for the current block
                threshold_range = range(np.min(block), np.max(block) + 1)
                criterias = np.array([self.compute_otsu_criteria(block, th) for th in threshold_range])
                best_threshold = threshold_range[np.argmin(criterias)]
                
                # Apply threshold to the current block and assign to the output image
                binary_block = np.zeros_like(block)
                binary_block[block >= best_threshold] = 255
                
                # Calculate block bounds for assignment
                block_height, block_width = binary_block.shape
                end_y = min(y + block_size, height)
                end_x = min(x + block_size, width)
                binary_image[y:end_y, x:end_x] = binary_block[:end_y-y, :end_x-x]
        
        return binary_image

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


    def local_optimal_thresholding(self, block_size = 5):
        # Convert image to grayscale
        gray_image = self.img
        
        # Get image dimensions
        height, width = gray_image.shape
        
        # Initialize the output binary image
        binary_image = np.zeros_like(gray_image, dtype=np.uint8)
        
        # Iterate over the image in blocks of specified size
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Define the current block within the image
                block = gray_image[y:y+block_size, x:x+block_size]
                
                # Apply optimal thresholding to the current block
                block_thresholded = self.apply_optimal_thresholding(block)
                
                # Assign the thresholded block to the corresponding region in the binary image
                block_height, block_width = block_thresholded.shape
                binary_image[y:y+block_height, x:x+block_width] = block_thresholded
        
        return binary_image

    def apply_optimal_thresholding(self, block):
        # Check if block is empty (all pixels are the same)
        if np.all(block == block[0, 0]):
            return np.zeros_like(block, dtype=np.uint8)  # Return all zeros for empty block
        
        # Compute the threshold using mean intensity of the block
        threshold = np.mean(block)
        
        # Iterate until convergence (threshold value stabilizes)
        while True:
            # Classify pixels into foreground (class 1) and background (class 2) based on current threshold
            foreground_pixels = block[block > threshold]
            background_pixels = block[block <= threshold]
            
            # Check if any class is empty
            if len(foreground_pixels) == 0 or len(background_pixels) == 0:
                return np.zeros_like(block, dtype=np.uint8)  # Return all zeros if any class is empty
            
            # Calculate mean intensity values of the two classes
            mean_foreground = np.mean(foreground_pixels)
            mean_background = np.mean(background_pixels)
            
            # Calculate new threshold as the average of mean intensities
            new_threshold = (mean_foreground + mean_background) / 2.0
            
            # Check convergence: if new threshold is close to the old threshold, break the loop
            if np.abs(new_threshold - threshold) < 1e-3:
                break
            
            # Update the threshold
            threshold = new_threshold
        
        # Apply the final threshold to the block
        block_thresholded = (block > threshold).astype(np.uint8) * 255
        
        return block_thresholded
    
    
    
    def local_multilevel_otsu_thresholding(self, num_classes, patch_size):
        """Segments the image using multilevel otsu thresholding in local patches

        Args:
            image (np.ndarray): input image (accepts grayscale only)  
            num_classes (int): number of classes to segment the image into 
            patch_size (int): pixel size of the patches used to divide the image

        Returns:
            np.ndarray: the segmented image.
        """
        image = self.img

        # Get image dimensions
        height = image.shape[0]
        width = image.shape[1]
        # print(gray_image.shape)
        
        # Initialize segmented image
        segmented_image = np.zeros_like(image)
        
        # Iterate over image patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                # Get the current patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Apply multilevel otsu thresholding to the patch
                patch_segmented, _ = self.multilevel_otsu_thresholding(patch, num_classes)
                
                # Assign segmented patch to the corresponding region in the segmented image
                segmented_image[y:y+patch_size, x:x+patch_size] = patch_segmented
        
        return segmented_image

    
    def multilevel_otsu_thresholding(self, num_classes):
        """Calculates Multi-Otsu Thresholds and returns the segmented image

        Args:
            image (np.ndarray): the input image. accepts greyscale only
            num_classes (int): number of classes to be segmented

        Returns:
            np.ndarray: the segmented image.
        """               
        image = self.img
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist_norm = hist.ravel() / hist.sum()
        
        # Compute cumulative sum of probabilities
        cumsum = np.cumsum(hist_norm)
        
        # Compute Otsu threshold for each class
        thresholds = np.zeros(num_classes - 1)
        for i in range(num_classes - 1):
            max_var, best_thresh = 0, 0
            for t in range(i * 256 // num_classes, (i+1) * 256 // num_classes):
                # Class probabilities
                w0 = cumsum[t] if t > 0 else 0
                w1 = cumsum[-1] - w0
                
                # Class means
                mu0 = np.sum(np.arange(0, t+1) * hist_norm[:t+1]) / (w0 + 1e-5)
                mu1 = np.sum(np.arange(t+1, 256) * hist_norm[t+1:]) / (w1 + 1e-5)
                
                # Class variances
                var = w0 * w1 * ((mu0 - mu1) ** 2)
                
                if var > max_var:
                    max_var = var
                    best_thresh = t
            thresholds[i] = best_thresh
        
        # Apply thresholding
        thresholds = np.sort(np.concatenate(([0], thresholds, [255])))
        segmented_image = np.digitize(image, thresholds)
        
        return segmented_image


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

# ######################################### K_means start ##################################################
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

# ######################################### K_means end ##################################################

# ######################################### agglomerative start ##################################################

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# def clusters_distance(cluster1, cluster2):
#     return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])


def clusters_distance_2(cluster1, cluster2):
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:

    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k

    def initial_clusters(self, points):
        initial_groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            initial_groups[(j, j, j)] = []
        for i, p in enumerate(points):
            go = min(initial_groups.keys(), key=lambda c: euclidean_distance(p, c))
            initial_groups[go].append(p)
        return [g for g in initial_groups.values() if len(g) > 0]

    def fit(self, points):

        # initially, assign each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)

        while len(self.clusters_list) > self.k:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            self.clusters_list.append(merged_cluster)

        # final cluster dictionary
        self.cluster = {}

        # Iterate over pixels in each cluster and give it the number of it is cluster
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):
        return self.cluster[tuple(point)]

    # take the pixel and return the new value of this pixel, first we know what it is cluster number and return
    # the average of this cluster as the new value of this pixel
    def predict_center(self, point):
        point_cluster_number = self.predict_cluster(point)
        center = self.centers[point_cluster_number]
        return center
# ######################################### agglomerative end ##################################################
