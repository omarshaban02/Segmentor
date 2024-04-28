import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import random


class WorkerSignals(QObject):
    get_segmented_image = pyqtSignal(np.ndarray)


class WorkerThread(QThread):
    def __init__(self, input_image, seeds, threshold):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image
        self.threshold = threshold
        self.seeds = seeds

    def run(self):
        segmented_img = region_growing(self.input_image, seeds=self.seeds, threshold=self.threshold)
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
