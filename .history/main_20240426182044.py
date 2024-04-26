import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QShortcut, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from home import Ui_MainWindow
import pyqtgraph as pg
# from classes import Image, WorkerThread, HoughTransform
import cv2
import json
from PyQt5.uic import loadUiType
import numpy as np

ui, _ = loadUiType("home.ui")


class Application(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.loaded_image = None

        self.gray_scale_image = None
        self.contour_thread = None
        
        self.scatter_item = pg.ScatterPlotItem(pen="lime", brush="lime", symbol="x", size=20)
        self.contour_line_item = pg.PlotDataItem(pen={'color': "r", 'width': 2})
        self.contour_line_item.setZValue(-1)

        self.initial_contour_points = []

        self.wgt_region_input.addItem(self.scatter_item)
        self.wgt_region_input.addItem(self.contour_line_item)

        self.actionOpen_Image.triggered.connect(self.open_image)

        # List containing all plotwidgets for ease of access
        self.plotwidget_set = [self.wgt_region_input, self.wgt_region_output,]


        # Create an image item for each plot-widget
        self.image_item_set = [self.item_region_input, self.item_region_output
                               ] = [pg.ImageItem() for _ in range(2)]

        # Initializes application components
        self.init_application()



        # self.btn_region_start.clicked.connect(self.process_image)


        ############################################ Connections ###################################################
        self.wgt_region_input.scene().sigMouseClicked.connect(self.on_mouse_click)
        



        #############################################################################################################
        self.undo_shortcut = QApplication.instance().installEventFilter(self)

        
    
    ################################## Initial Contour Handling Section #########################################
    # Event filter to handle pressing Ctrl + Z to undo initial contour
    def eventFilter(self, source, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Z and QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.undo_last_point()
            return True

        if event.type() == event.KeyPress and event.key() == Qt.Key_S and QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.save_chain_code()
            return True
        return super().eventFilter(source, event)

    # Handles clicking on contour input display widget
    def on_mouse_click(self, event):

        # Allows for checking if a keyboard modifier is pressed, ex: Ctrl
        modifiers = QApplication.keyboardModifiers()

        if event.button() == 1:
            clicked_point = self.wgt_region_input.plotItem.vb.mapSceneToView(event.scenePos())
            print(f"Mouse clicked at {clicked_point}")

            point_x = clicked_point.x()
            point_y = clicked_point.y()

            self.initial_contour_points.append((point_x, point_y))
            # self.scatter_item.addPoints(x=[ev.scenePos().x()], y=[ev.scenePos().y()])
            self.scatter_item.addPoints(x=[clicked_point.x()], y=[clicked_point.y()])
            self.contour_line_item.setData(
                x=[p[0] for p in self.initial_contour_points + [self.initial_contour_points[0]]],
                y=[p[1] for p in self.initial_contour_points + [self.initial_contour_points[0]]])

            if modifiers == Qt.ControlModifier:
                self.clear_points()

    def undo_last_point(self):

        if self.initial_contour_points:
            self.initial_contour_points.pop()

            # Update the scatter item
            self.scatter_item.setData(x=[p[0] for p in self.initial_contour_points],
                                      y=[p[1] for p in self.initial_contour_points])

            # Update the line item
            if len(self.initial_contour_points) > 1:
                self.contour_line_item.setData(
                    x=[p[0] for p in self.initial_contour_points + [self.initial_contour_points[0]]]
                    , y=[p[1] for p in self.initial_contour_points + [self.initial_contour_points[0]]])
            else:
                self.contour_line_item.clear()

    def clear_points(self):
        self.initial_contour_points = []
        # Clear scatter plot
        self.scatter_item.clear()
        # Clear line plot
        self.contour_line_item.clear()

    ################################## END Initial Contour Handling Section #########################################

    def update_contour_image(self, image):
        self.display_image(self.item_region_output, image)

    def processing_finished(self):
        print("Processing Finished")

    def update_area_perimeter(self, area, perimeter):
        self.lbl_area.setText(f"{round(area, 2)}")
        self.lbl_perimeter.setText(f"{round(perimeter, 2)}")

    # def process_image(self):
    #     self.contour_thread = WorkerThread(self.gray_scale_image, self.initial_contour_points)
    #     self.contour_thread.signals.update.connect(self.update_contour_image)
    #     self.contour_thread.signals.finished.connect(self.processing_finished)
    #     self.contour_thread.signals.calc_area_perimeter.connect(self.update_area_perimeter)
    #     self.contour_thread.start()
 



    # ############################### Misc Functions ################################

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.setZValue(-2)
        image_item.getViewBox().autoRange()

    def load_img_file(self, image_path):
        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        
        self.display_image(self.item_region_input, self.loaded_image)
        

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
        # self.setup_hough_sliders()
        # self.setup_checkboxes()


app = QApplication(sys.argv)
win = Application()  
win.show()
app.exec()
