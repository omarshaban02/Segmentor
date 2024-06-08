import numpy as np


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
