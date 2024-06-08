from PyQt5.QtCore import QThread, pyqtSignal, QObject
import numpy as np
from classes.RegionGrowing import *
from classes.MeanShift import mean_shift

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
        segmented_img = RegionGrowing(self.input_image, seeds=self.seeds, threshold=self.threshold)
        self.signals.get_segmented_image.emit(segmented_img)


class MeanShiftThread(QThread):
    def __init__(self, input_image):
        super(MeanShiftThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image

    def run(self):
        segmented_img = mean_shift(self.input_image)
        self.signals.get_segmented_image.emit(segmented_img)
