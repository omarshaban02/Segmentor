import numpy as np
import cv2


class Thresholding:
    def __init__(self, input_image):
        self.img = input_image.copy()

    def global_threshold(self, threshold_value=127, max_value=255):
        """
        Apply global thresholding to a grayscale image.

        Parameters:
            threshold_value (int): Threshold value.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Threshold image.
        """
        image = self.img
        thresholded = np.where(image > threshold_value, max_value, 0).astype(np.uint8)
        return thresholded

    def local_threshold(self, block_size=11, c=2, max_value=255):
        """
        Apply local thresholding to a grayscale image.

        Parameters:
            block_size (int): Size of the local neighborhood for computing the threshold value.
            c (int): Constant subtracted from the mean or weighted mean.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Thresholded image.
        """
        image = self.img
        thresholded = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Define the region of interest
                roi = image[max(0, i - block_size // 2): min(image.shape[0], i + block_size // 2),
                      max(0, j - block_size // 2): min(image.shape[1], j + block_size // 2)]
                # Compute the threshold value for the region
                threshold_value = np.mean(roi) - c
                # Apply thresholding
                thresholded[i, j] = max_value if image[i, j] > threshold_value else 0
        return thresholded

    def compute_otsu_criteria(self, img, th):
        # create the threshold image
        thresholded_im = np.zeros(img.shape)
        thresholded_im[img >= th] = 1

        # compute weights
        nb_pixels = img.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        # if one of the classes is empty, ex: all pixels are below or above the threshold,
        # that threshold will not be considered
        # in the search for the best threshold
        if weight1 == 0 or weight0 == 0:
            return np.inf

        # find all pixels belonging to each class
        val_pixels1 = img[thresholded_im == 1]
        val_pixels0 = img[thresholded_im == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

        return weight0 * var0 + weight1 * var1

    def otsu_thresholding(self):
        img = self.img
        threshold_range = range(np.max(img) + 1)
        criteria = np.array([self.compute_otsu_criteria(img, th) for th in threshold_range])

        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criteria)]

        binary = img
        binary[binary < best_threshold] = 0
        binary[binary >= best_threshold] = 255

        return binary

    def local_otsu_thresholding(self, block_size=5, sigma=1):
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
                block = blurred_image[y:y + block_size, x:x + block_size]

                # Calculate the Otsu threshold for the current block
                threshold_range = range(np.min(block), np.max(block) + 1)
                criteria = np.array([self.compute_otsu_criteria(block, th) for th in threshold_range])
                best_threshold = threshold_range[np.argmin(criteria)]

                # Apply threshold to the current block and assign to the output image
                binary_block = np.zeros_like(block)
                binary_block[block >= best_threshold] = 255

                # Calculate block bounds for assignment
                block_height, block_width = binary_block.shape
                end_y = min(y + block_size, height)
                end_x = min(x + block_size, width)
                binary_image[y:end_y, x:end_x] = binary_block[:end_y - y, :end_x - x]

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

    def local_optimal_thresholding(self, block_size=5):
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
                block = gray_image[y:y + block_size, x:x + block_size]

                # Apply optimal thresholding to the current block
                block_thresholded = self.apply_optimal_thresholding(block)

                # Assign the thresholded block to the corresponding region in the binary image
                block_height, block_width = block_thresholded.shape
                binary_image[y:y + block_height, x:x + block_width] = block_thresholded

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
        block_threshold = (block > threshold).astype(np.uint8) * 255

        return block_threshold

    def local_multilevel_otsu_thresholding(self, num_classes=3, patch_size=100):
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

        # Initialize segmented image
        segmented_image = np.zeros_like(image)

        # Iterate over image patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                # Get the current patch
                patch = image[y:y + patch_size, x:x + patch_size]

                # Apply multilevel otsu thresholding to the patch
                patch_segmented = self.multilevel_otsu_thresholding(patch, num_classes)

                # Assign segmented patch to the corresponding region in the segmented image
                segmented_image[y:y + patch_size, x:x + patch_size] = patch_segmented

        return segmented_image

    def multilevel_otsu_thresholding(self, image, num_classes):
        """Calculates Multi-Otsu Thresholds and returns the segmented image

        Args:
            image (np.ndarray): the input image. accepts greyscale only
            num_classes (int): number of classes to be segmented

        Returns:
            np.ndarray: the segmented image.
        """

        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()

        # Compute cumulative sum of probabilities
        cumsum = np.cumsum(hist_norm)

        # Compute Otsu threshold for each class
        thresholds = np.zeros(num_classes - 1)
        for i in range(num_classes - 1):
            max_var, best_thresh = 0, 0
            for t in range(i * 256 // num_classes, (i + 1) * 256 // num_classes):
                # Class probabilities
                w0 = cumsum[t] if t > 0 else 0
                w1 = cumsum[-1] - w0

                # Class means
                mu0 = np.sum(np.arange(0, t + 1) * hist_norm[:t + 1]) / (w0 + 1e-5)
                mu1 = np.sum(np.arange(t + 1, 256) * hist_norm[t + 1:]) / (w1 + 1e-5)

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
