import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QShortcut, QSlider
from PyQt5.QtCore import Qt
from home import Ui_MainWindow
import pyqtgraph as pg
import cv2
from PyQt5.uic import loadUiType
from classes import WorkerThread

ui, _ = loadUiType("home.ui")


class Application(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.loaded_image = None
        self.loaded_image_gray = None

        self.gray_scale_image = None
        self.contour_thread = None

        self.scatter_item = pg.ScatterPlotItem(pen="lime", brush="lime", symbol="x", size=20)

        self.initial_region_seeds = []
        self.region_growing_thread = None

        self.wgt_seg_input.addItem(self.scatter_item)

        self.actionOpen_Image.triggered.connect(self.open_image)

        # List containing all plotwidgets for ease of access
        self.plotwidget_set = [self.wgt_seg_input, self.wgt_seg_output, self.wgt_thresh_input, self.wgt_thresh_output]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_seg_input, self.item_seg_output, self.item_thresh_input
                               ,self.item_thresh_output] = [pg.ImageItem() for _ in range(4)]

        # Initializes application components
        self.init_application()

        self.btn_seg_apply.clicked.connect(self.process_image)

        ############################################ Connections ###################################################
        self.wgt_seg_input.scene().sigMouseClicked.connect(self.on_mouse_click)

        #############################################################################################################
        self.undo_shortcut = QApplication.instance().installEventFilter(self)

    ################################## Initial Contour Handling Section #########################################
    # Event filter to handle pressing Ctrl + Z to undo initial contour
    def eventFilter(self, source, event):
        """
        Just an event filter to capture "Ctrl + Z" combination to undo the last point
        
        """

        if event.type() == event.KeyPress and event.key() == Qt.Key_Z and QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.undo_last_point()
            return True

        return super().eventFilter(source, event)

    def on_mouse_click(self, event):
        """
        Handles clicking the region input plot to add region seeds
        """

        # Allows for checking if a keyboard modifier is pressed, ex: Ctrl
        modifiers = QApplication.keyboardModifiers()

        if event.button() == 1:
            clicked_point = self.wgt_seg_input.plotItem.vb.mapSceneToView(event.scenePos())
            print(f"Mouse clicked at {clicked_point}")

            point_x = clicked_point.x()
            point_y = clicked_point.y()

            self.initial_region_seeds.append((point_x, point_y))
            self.scatter_item.addPoints(x=[clicked_point.x()], y=[clicked_point.y()])

            if modifiers == Qt.ControlModifier:
                self.clear_points()

    def undo_last_point(self):
        """
        undos the last added region seed
        
        """

        if self.initial_region_seeds:
            self.initial_region_seeds.pop()

            # Update the scatter item
            self.scatter_item.setData(x=[p[0] for p in self.initial_region_seeds],
                                      y=[p[1] for p in self.initial_region_seeds])

    def clear_points(self):
        self.initial_region_seeds = []

        # Clear scatter plot
        self.scatter_item.clear()

    ################################## END Initial Contour Handling Section #########################################

    def update_region_growing_output(self, segmented_image):
        self.display_image(self.item_seg_output, segmented_image)

    def process_image(self):
        current_seg_mode = self.comboBox_seg_mode.currentIndex()
        self.region_growing_thread = WorkerThread(self.loaded_image_gray, list(
            map(lambda tpl: (int(tpl[0]), int(tpl[1])), self.initial_region_seeds)),
                                                  self.sld_region_threshold.value())
        self.region_growing_thread.signals.get_segmented_image.connect(self.update_region_growing_output)
        self.region_growing_thread.start()

    # ############################### Misc Functions ################################

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.setZValue(-2)
        image_item.getViewBox().autoRange()

    def load_img_file(self, image_path):
        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        self.loaded_image_gray = cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2GRAY)
        
        # Displays the image on both tabs in the input widget
        for item in (self.item_seg_input, self.item_thresh_input):
            self.display_image(item, self.loaded_image)

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file)

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            # Removes Axes and Padding from all plotwidgets intended to display an image
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotwidget.setBackground((25, 30, 40))
            plotitem = plotwidget.getPlotItem()
            plotitem.getViewBox().setDefaultPadding(0)
            plotitem.showGrid(True)
            plotwidget.setMouseEnabled(x=False, y=False)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)

    def init_application(self):
        self.setup_plotwidgets()


app = QApplication(sys.argv)
win = Application()
win.show()
app.exec()
