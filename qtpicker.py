
import sys

# Paths, suppress annoying Qt warning
import os
os.environ['QT_LOGGING_RULES'] = 'qt.pointer.dispatch=false'
from glob import glob

# Arrays
import numpy as np

# PySide tools
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtGui import Qt as QtGui_Qt

# pyqtgraph
import pyqtgraph
pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
from pyqtgraph import ImageView, ImageItem, PolyLineROI, mkColor, PlotDataItem

# quot tools
from quot.read import ImageReader
from quot.helper import get_edges, get_ordered_mask_points


class ClickableMask(ImageItem):
    def __init__(self, parent):
        self.parent = parent
        self.points = parent.points
        self.mask = parent.mask

        lut = np.zeros((len(parent.colors), 3))
        color = mkColor(parent.curr_color)
        lut[1] = (color.red(), color.green(), color.blue())

        super().__init__(self.mask, opacity=parent.opacity, lut=lut)


    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pos = self.mapFromScene(event.scenePos())
            print(pos)
            y, x = pos.x(), pos.y()
            if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
                if self.mask[int(x), int(y)]:
                    self.on_mask_clicked()

    def on_mask_clicked(self):
        self.parent.toggle_label()


class EditableMask(PolyLineROI):
    def __init__(self, parent):
        super().__init__(parent.points, closed=True)
        #self.mask = parent.mask
        #self.points = parent.points
        #self.opacity = parent.opacity


class ClickableEditableLabeledMask:
    def __init__(self, 
                 mask: np.ndarray, 
                 imv: ImageView=None, 
                 possible_labels: list[str]=['good', 'bad'],
                 colors: list[str]=['g', 'r'],
                 clickable: bool=True, 
                 opacity: float=0.1):
        
        self.mask = mask
        self.imv = imv
        self.opacity = opacity
        self.clickable = clickable
        self.points = get_ordered_mask_points(get_edges(mask))
        # Reverse points to match pyqtgraph's row-major order
        self.points = np.flip(self.points, axis=1)

        self.idx = 0
        self.possible_labels = possible_labels
        self.colors = colors
        self.curr_label = self.possible_labels[self.idx]
        self.curr_color = self.colors[self.idx]

        self.clickable_mask = ClickableMask(self)
        self.editable_mask = EditableMask(self)

    def add_to_imv(self, imv=None):
        """Add the mask to the image view, overwriting self.imv if provided."""
        if imv is not None:
            self.imv = imv
        if self.imv is not None:
            if self.clickable:
                self.imv.addItem(self.clickable_mask)
            else:
                self.imv.addItem(self.editable_mask)
    
    def remove_from_imv(self, imv=None):
        """Remove the mask from the image view, overwriting self.imv if provided."""
        if imv is not None:
            self.imv = imv
        if self.imv is not None:
            if self.clickable:
                self.imv.removeItem(self.clickable_mask)
            else:
                self.imv.removeItem(self.editable_mask)
    
    def toggle_editable(self):
        self.remove_from_imv()
        self.clickable = not self.clickable
        self.add_to_imv()
    
    def toggle_label(self):
        self.remove_from_imv()
        self.idx = (self.idx + 1) % len(self.possible_labels)
        self.curr_label = self.possible_labels[self.idx]
        self.curr_color = self.colors[self.idx]
        self.clickable_mask = ClickableMask(self)
        self.editable_mask = EditableMask(self)
        self.add_to_imv()
    
        
class ImageGrid(QWidget):
    """
    path       :   str, path to automation folder
    """
    def __init__(self, path, shape, show_image_histogram=True, parent=None):
        super(ImageGrid, self).__init__(parent=parent)
        self.path = path
        self.grid_shape = shape
        self.show_hists = show_image_histogram

        self.init_data()
        self.init_UI()
    
    def init_data(self):
        """Read in masks and images, match them up to each other, and
        coerce them into an array to be displayed in the GUI."""
        self.snaps_folders = glob(os.path.join(self.path, "snaps2*"))
        self.snaps_folders.sort()

        self.snaps = glob(os.path.join(self.snaps_folders[0], "*.tif*"))
        self.n_snaps = len(self.snaps)
        self.n_windows = int(np.ceil(self.n_snaps / (self.grid_shape[0] * self.grid_shape[1])))
        self.image_shape = ImageReader(self.snaps[0]).shape
        self.sorted_data = np.zeros((self.n_windows * self.grid_shape[0] * self.grid_shape[1],
                                     len(self.snaps_folders),
                                     self.image_shape[1],
                                     self.image_shape[2]))
        self.masks = np.ndarray(shape=self.sorted_data.shape[0], dtype=object)
        self.masks.fill([])

        if os.path.exists(os.path.join(self.path, 'rois.txt')):
            self.rois = np.loadtxt(os.path.join(self.path, "rois.txt"), delimiter=',')
            pad = self.sorted_data.shape[0] - self.rois.shape[0]
            self.rois = np.pad(self.rois, ((0, pad), (0, 0)))
        else:
            self.rois = None

        # Loop over snaps folders
        for i in range(self.n_snaps):
            for j, folder in enumerate(self.snaps_folders):
                # Load image into array
                filename = os.path.join(folder, f"{i+1}.tif")
                if os.path.exists(filename):
                    self.sorted_data[i, j] = ImageReader(filename).get_frame(0)

            # Get mask corresponding to this image and read in
            mask_filename = os.path.join(self.path, "masks", f"{i+1}.csv")
            if os.path.exists(mask_filename):
                mask = np.loadtxt(mask_filename, delimiter=',')
                # Create a mask for each non-zero value
                self.masks[i] = [ClickableEditableLabeledMask(mask == v, opacity=0.1) \
                                 for v in np.unique(mask[mask > 0])]

        # Reshape everything to (n_windows, grid_shape[0], grid_shape[1], ...)
        self.sorted_data = np.reshape(self.sorted_data, 
                                      newshape=(self.n_windows, 
                                                self.grid_shape[0], 
                                                self.grid_shape[1],
                                                len(self.snaps_folders),
                                                self.image_shape[1], 
                                                self.image_shape[2]))
        self.masks = np.reshape(self.masks, 
                                newshape=(self.sorted_data.shape[:3]))
        if self.rois is not None:
            self.rois = np.reshape(self.rois,
                                   newshape=(*self.sorted_data.shape[:3], 4))
    
    def init_UI(self):
        """Initialize the user interface."""
        self.window = QWidget()
        layout = QGridLayout(self.window)

        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.window)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.window)
        self.left_shortcut.activated.connect(self.prev_window)
        self.right_shortcut.activated.connect(self.next_window)

        self.window_idx = 0
        self.image_views = np.zeros((self.grid_shape[0], self.grid_shape[1]), 
                                    dtype=object)
        self.roi_views = np.zeros((self.grid_shape[0], self.grid_shape[1]),
                                    dtype=object)
        for (i, j), _ in np.ndenumerate(self.image_views):
            # Make image views
            self.image_views[i, j] = ImageView(parent=self.window)
            layout.addWidget(self.image_views[i, j], i, j)

            # Hide buttons            
            self.image_views[i, j].ui.roiBtn.hide()
            self.image_views[i, j].ui.menuBtn.hide()
            if not self.show_hists:
                self.image_views[i, j].ui.histogram.hide()

            # Add ROI objects
            self.roi_views[i, j] = PlotDataItem(x=[], y=[])
            self.image_views[i, j].addItem(self.roi_views[i, j])

        # Populate image views
        self.update_window()

        # Resize main window
        self.window.resize(1280, 720)
        self.window.show()
    
    def next_window(self):
        """Go to the next grid of images."""
        self.remove_masks()
        if self.window_idx < self.n_windows:
            self.window_idx += 1
            self.update_window()

    def prev_window(self):
        """Go to the previous grid of images."""
        self.remove_masks()
        if self.window_idx > 0:
            self.window_idx -= 1
            self.update_window()
    
    def remove_masks(self):
        """Save the masks remove them from their respective ImageViews."""
        for (i, j), _ in np.ndenumerate(self.image_views):
            for mask_item in self.masks[self.window_idx, i, j]:
                mask_item.remove_from_imv()
    
    def update_window(self):
        """Update the current window."""
        # Make sure window index is valid
        if self.window_idx < 0:
            self.window_idx = 0
        elif self.window_idx >= self.n_windows:
            self.window_idx = self.n_windows - 1
        
        for (i, j), _ in np.ndenumerate(self.image_views):
            self.image_views[i, j].setImage(self.sorted_data[self.window_idx, i, j, 0])

            for mask_item in self.masks[self.window_idx, i, j]:
                mask_item.add_to_imv(self.image_views[i, j])
            
            # Add ROI as a red box
            if self.rois is not None:
                self.roi_views[i, j].setData(y=[self.rois[self.window_idx, i, j, 1],
                                                self.rois[self.window_idx, i, j, 3],
                                                self.rois[self.window_idx, i, j, 3],
                                                self.rois[self.window_idx, i, j, 1],
                                                self.rois[self.window_idx, i, j, 1]],
                                             x=[self.rois[self.window_idx, i, j, 0],
                                                self.rois[self.window_idx, i, j, 0],
                                                self.rois[self.window_idx, i, j, 2],
                                                self.rois[self.window_idx, i, j, 2],
                                                self.rois[self.window_idx, i, j, 0]],
                                                pen='r')
            else:
                self.roi_views[i, j].setData(x=[], y=[], pen='r')


if __name__ == '__main__':
    folder = str(input("Drag output automation folder: ").strip())
    app = QApplication(sys.argv)
    window = ImageGrid(folder, shape=(2, 4))
    window.show()
    app.exec()