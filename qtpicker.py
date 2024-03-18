"""
qtpicker.py: A GUI for picking cells in a grid of images.
"""
# Paths, suppress annoying Qt warning
import os
os.environ['QT_LOGGING_RULES'] = 'qt.pointer.dispatch=false'
from glob import glob

# Arrays, DataFrames
import numpy as np
import pandas as pd

# PySide tools
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication, QPushButton
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtGui import Qt as QtGui_Qt

# Plotting, colors
import pyqtgraph
pyqtgraph.setConfigOptions(imageAxisOrder='row-major')  # We use (x, y) indexing
from pyqtgraph import ImageView, ImageItem, PolyLineROI, mkColor, PlotDataItem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 

# Image tools
import tifffile
from quot.read import ImageReader
from scipy import ndimage
from skimage.measure import inertia_tensor_eigvals

# Masking tools
from quot.helper import get_edges, get_ordered_mask_points
from quot.gui.masker import apply_masks
from matplotlib.path import Path 

# Progress bar
from tqdm import tqdm


VERSION = "0.1.0"


###############
## UTILITIES ##
###############
def create_enclosed_mask(points: list, shape: tuple) -> np.ndarray:
    """
    Create an enclosed Boolean mask from a list of points.

    args:
    -----
    points  :   list or ndarray of shape (2, n), 
                where each point is a tuple of (x, y).
    shape   :   tuple of (height, width).

    return:
    -------
    mask    :   ndarray of type bool
    """
    y, x = np.indices(shape)
    coordinates = np.column_stack((x.flatten(), y.flatten()))
    mask = Path(points).contains_points(coordinates).reshape(shape)
    return mask

def create_integer_mask(point_sets, shape):
    """
    Create an enclosed integer mask from a list of a list points.

    args:
    -----
    point_sets  :   list of (list or ndarray of shape (2, n)),
                    where each point is a tuple of (x, y).
    shape       :   tuple of (height, width).

    return:
    -------
    out_mask    :   ndarray of int, where the value at each coordinate
                    is the index + 1 of the point set that contains it.
                    Pixels outside all masks are 0.
    """
    y, x = np.indices(shape)
    coordinates = np.column_stack((x.flatten(), y.flatten()))
    out_mask = np.zeros(shape, dtype=int)
    for i, points in enumerate(point_sets):
        mask = Path(points, closed=True).contains_points(coordinates).reshape(shape)
        out_mask[mask] = i+1
    return out_mask

def get_mask_points(mask, max_points=25):
    """Wrapper for quot.helper.get_ordered_mask_points, 
    returns flipped points to match row-major order."""
    return np.flip(get_ordered_mask_points(mask, max_points=max_points), 
                   axis=1)

def get_eccentricity(binary_mask: np.ndarray):
    """Taken from skimage:
    Eccentricity of the ellipse that has the same second-moments as the
    region. The eccentricity is the ratio of the focal distance
    (distance between focal points) over the major axis length.
    The value is in the interval [0, 1).
    When it is 0, the ellipse becomes a circle."""
    l1, l2 = inertia_tensor_eigvals(binary_mask)
    if l1 == 0:
        return 0
    return np.sqrt(1 - l2 / l1)

##################
## MASK CLASSES ##
##################
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
            y, x = pos.x(), pos.y()
            if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
                if self.mask[int(x), int(y)]:
                    self.on_mask_clicked()

    def on_mask_clicked(self):
        self.parent.cycle_label()
        print(self.parent.curr_label)


class EditableMask(PolyLineROI):
    def __init__(self, parent):
        super().__init__(parent.points, closed=True)
        self.parent = parent
        self.sigRegionChanged.connect(self.mask_changed)

    def mask_changed(self):
        self.parent.points = np.asarray([[self.mapSceneToParent(p[1]).x(), 
                                          self.mapSceneToParent(p[1]).y()] \
                                         for p in super().getSceneHandlePositions()])
        self.parent.mask = create_enclosed_mask(self.parent.points, 
                                                self.parent.shape)


class ClickableEditableLabeledMask:
    def __init__(self, 
                 mask: np.ndarray,
                 idx: int=0, 
                 imv: ImageView=None, 
                 possible_labels: list[str]=['good', 'bad'],
                 colors: list[str]=['g', 'r'],
                 clickable: bool=True,
                 shown: bool=False, 
                 opacity: float=0.15):
        
        self.mask = mask
        self.shape = mask.shape
        self.imv = imv
        self.opacity = opacity
        self.clickable = clickable
        self.points = get_mask_points(get_edges(mask))

        self.idx = idx
        self.possible_labels = possible_labels
        self.colors = colors
        self.curr_label = self.possible_labels[self.idx]
        self.curr_color = self.colors[self.idx]
        self.shown = shown

        self.clickable_mask = ClickableMask(self)
        self.editable_mask = EditableMask(self)

    def add_to_imv(self, imv=None):
        """Add the mask to the image view, 
        overwriting self.imv if provided."""
        # Don't do anything if mask is already shown
        if self.shown:
            return
        # Override self.imv if provided
        if imv is not None:
            self.imv = imv
        # Add mask to image view
        if self.imv is not None:
            if self.clickable:
                self.imv.addItem(self.clickable_mask)
            # Bad masks that are editable do not get added
            elif not self.clickable and self.curr_label != 'bad':
                self.imv.addItem(self.editable_mask)
            else:
                return
        self.shown = True
    
    def remove_from_imv(self, imv=None):
        """Remove the mask from the image view, 
        overwriting self.imv if provided."""
        # Don't do anything if mask is already removed
        if not self.shown:
            return
        # Override self.imv if provided
        if imv is not None:
            self.imv = imv
        if self.imv is not None:
            if self.clickable:
                self.imv.removeItem(self.clickable_mask)
            else:
                self.imv.removeItem(self.editable_mask)
        self.shown = False
    
    def toggle_editable(self):
        """Change between a clickable and an editable mask."""
        if self.shown:
            self.remove_from_imv()

        # Update ClickableMask in case points are changed
        if not self.clickable:
            self.clickable_mask = ClickableMask(self)        
        
        self.clickable = not self.clickable
      
        self.add_to_imv()

    def cycle_label(self):
        self.remove_from_imv()
        self.idx = (self.idx + 1) % len(self.possible_labels)
        self.curr_label = self.possible_labels[self.idx]
        self.curr_color = self.colors[self.idx]
        self.clickable_mask = ClickableMask(self)
        self.editable_mask = EditableMask(self)
        self.add_to_imv()
    
###############
## GUI CLASS ##
###############        
class ImageGrid(QWidget):
    """
    path                    :   str, path to automation folder
    shape                   :   tuple[int, int], shape of grid
    show_image_histogram    :   bool, show LUT
    roi_masks_only          :   bool, only show masks within ROI,
                                which saves loading time
    mask_opacity            :   float, opacity of clickable masks
    parent                  :   root QWidget, if any
    """
    def __init__(self, 
                 path: str, 
                 shape: tuple[int, int],
                 show_image_histogram: bool=True,
                 save_mask_png: bool=False,
                 roi_masks_only: bool=False,
                 min_mask_area: int=0,
                 possible_labels: list[str]=['good', 'bad'],
                 colors: list[str]=['g', 'r'],
                 mask_opacity: float=0.15, 
                 parent: object=None):
        super(ImageGrid, self).__init__(parent=parent)
        self.path = path
        self.grid_shape = shape
        self.n_images_displayed = shape[0] * shape[1]
        self.save_mask_png = save_mask_png
        self.show_hists = show_image_histogram
        self.roi_masks_only = roi_masks_only
        self.opacity = mask_opacity
        self.min_mask_area = min_mask_area
        self.possible_labels = possible_labels
        self.colors = colors
        self.freestyle_mode = False
        self.edit_mode = False
        self.masks_shown = True

        self.init_data()
        self.init_UI()
    
    def init_data(self):
        """Read in masks and images and  match them up 
        to each other to be displayed in the UI"""
        self.snaps_folders = sorted(glob(os.path.join(self.path, "snaps2*")))
        self.snaps = glob(os.path.join(self.snaps_folders[0], "*.tif*"))
        self.n_snaps = len(self.snaps)
        self.n_windows = int(np.ceil(self.n_snaps / self.n_images_displayed))
        self.masks = np.ndarray(shape=self.n_windows * self.n_images_displayed, 
                                dtype=object)

        if os.path.exists(os.path.join(self.path, 'rois.txt')):
            self.rois = np.loadtxt(os.path.join(self.path, "rois.txt"), 
                                   delimiter=',', 
                                   dtype=int)
        else:
            self.rois = None
            print("Warning: no ROIs found. No masks will be applied to this dataset.")

        # Loop over snap images, snap folders to populate filepaths
        self.snaps_filepaths = np.full(shape=(len(self.snaps_folders), 
                                                 self.n_snaps),
                                       fill_value=np.nan, 
                                       dtype=object)
        for j in tqdm(range(self.n_snaps), "Loading images and masks"):
            for i, folder in enumerate(self.snaps_folders):
                # Load image into array
                filename = os.path.join(folder, f"{j+1}.tif")
                if os.path.exists(filename):
                    self.snaps_filepaths[i, j] = filename

                # Get mask corresponding to this image and read in
                mask_filename = os.path.join(self.path, "masks", f"{j+1}.csv")
                if not os.path.exists(mask_filename):
                    self.masks[j] = []
                    continue

                # Load mask from CSV
                mask = np.loadtxt(mask_filename, delimiter=',')

                # Only look at masks inside ROI if specified
                if self.roi_masks_only and self.rois is not None:
                    # Get ROI for this image
                    roi = self.rois[j]
                    # Crop mask to ROI
                    crop = mask[roi[0]:roi[2], roi[1]:roi[3]]
                    # Get only unique indices within the ROI
                    idx, counts = np.unique(crop[crop > 0], return_counts=True)
                else:
                    # Else get all unique indices in the mask
                    idx, counts = np.unique(mask[mask > 0], return_counts=True)
                    
                # Create a mask for each non-zero value if over min area
                ma_indices = idx[counts > self.min_mask_area]
                self.masks[j] = [ClickableEditableLabeledMask(mask == v, 
                                                              opacity=self.opacity,
                                                              possible_labels=self.possible_labels,
                                                              colors=self.colors) \
                                for v in ma_indices]
    
    def init_UI(self):
        """Initialize the user interface."""
        self.window = QWidget()
        self.layout = QGridLayout(self.window)
        
        self.window_idx = 0
        self.channel_idx = 0
        self.n_channels = len(self.snaps_folders)
        self.image_views = np.zeros((self.grid_shape[0], self.grid_shape[1]), 
                                    dtype=object)
        self.roi_views = np.zeros((self.grid_shape[0], self.grid_shape[1]),
                                    dtype=object)
        for i, j in np.ndindex(self.image_views.shape):
            # Make image views
            self.image_views[i, j] = ImageView(parent=self.window)
            self.layout.addWidget(self.image_views[i, j], i, j)

            # Hide buttons            
            self.image_views[i, j].ui.roiBtn.hide()
            self.image_views[i, j].ui.menuBtn.hide()
            if not self.show_hists:
                self.image_views[i, j].ui.histogram.hide()

            # Add ROI objects
            self.roi_views[i, j] = PlotDataItem(x=[], y=[])
            self.image_views[i, j].addItem(self.roi_views[i, j])
        
        # Add buttons to the short edge of the grid
        n_buttons = 10
        # If there are fewer columns than rows, add buttons to the right
        if self.grid_shape[0] > self.grid_shape[1]:
            button_indices = [(i % self.grid_shape[0], 
                               i // self.grid_shape[0] + self.grid_shape[1]) 
                              for i in range(n_buttons)]
        # Else add buttons to the bottom
        else:
            button_indices = [(i // self.grid_shape[1] + self.grid_shape[0],
                               i % self.grid_shape[1]) 
                              for i in range(n_buttons)]

        # Add buttons to go through channels
        self.B_next_chan = QPushButton("Next channel (w, ↑)", self.window)
        self.B_prev_chan = QPushButton("Previous channel (s, ↓)", self.window)
        self.layout.addWidget(self.B_next_chan, *button_indices[0])
        self.layout.addWidget(self.B_prev_chan, *button_indices[1])

        # w shortcut to go to next channel, s to go to previous
        self.w_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_W), self.window)
        self.s_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_S), self.window)
        self.down_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Down), self.window)
        self.up_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Up), self.window)
        
        # Add buttons to advance windows
        self.B_prev = QPushButton("Previous window (a, ←)", self.window)
        self.B_next = QPushButton("Next window (d, →)", self.window)
        self.layout.addWidget(self.B_prev, *button_indices[2])
        self.layout.addWidget(self.B_next, *button_indices[3])

        # a and d key shortcuts to advance windows
        self.a_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_A), self.window)
        self.d_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_D), self.window)
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.window)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.window)

        # Add a button to toggle editable masks
        self.B_toggle_editable = QPushButton("Toggle editable (e)", self.window)
        self.B_toggle_editable.clicked.connect(self.toggle_editable)
        self.layout.addWidget(self.B_toggle_editable, *button_indices[4])

        # e key shortcut to toggle editable masks
        self.e_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_E), self.window)
        self.e_shortcut.activated.connect(self.toggle_editable)

        # Add button to toggle showing masks
        self.B_toggle_masks = QPushButton("Toggle masks (q)", self.window)
        self.B_toggle_masks.clicked.connect(self.toggle_masks)
        self.layout.addWidget(self.B_toggle_masks, *button_indices[5])

        # q key shortcut to toggle masks
        self.q_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Q), self.window)
        self.q_shortcut.activated.connect(self.toggle_masks)

        # Add a button to freestyle draw masks
        self.B_freestyle = QPushButton("Draw masks (r)", self.window)
        self.B_freestyle.clicked.connect(self.freestyle)
        self.layout.addWidget(self.B_freestyle, *button_indices[6])

        # r shortcut to draw masks
        self.r_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_R), self.window)
        self.r_shortcut.activated.connect(self.freestyle)

        # Add a button to finish and apply masks
        self.B_apply_masks = QPushButton("Save and apply masks", self.window)
        self.B_apply_masks.clicked.connect(self.apply_masks)
        self.layout.addWidget(self.B_apply_masks, *button_indices[7])

        # Add a button to pickle masks
        self.B_apply_masks = QPushButton("Save masking progress", self.window)
        self.B_apply_masks.clicked.connect(self.save_state)
        self.layout.addWidget(self.B_apply_masks, *button_indices[8])

        # Add a button to load pickled masks
        self.B_apply_masks = QPushButton("Load masking progress", self.window)
        self.B_apply_masks.clicked.connect(self.load_state)
        self.layout.addWidget(self.B_apply_masks, *button_indices[9])

        # Populate image views
        self.update_window()
        self.add_masks()

        # Resize main window and show
        self.connect_navigation_buttons_shortcuts()
        self.window.resize(1280, 720)
        self.window.show()        

    def connect_navigation_buttons_shortcuts(self):
        """Allow for navigation between windows and channels
        using buttons and keyboard shortcuts."""
        self.B_next_chan.clicked.connect(self.next_channel)
        self.B_prev_chan.clicked.connect(self.prev_channel)
        self.w_shortcut.activated.connect(self.next_channel)
        self.s_shortcut.activated.connect(self.prev_channel)
        self.down_shortcut.activated.connect(self.next_channel)
        self.up_shortcut.activated.connect(self.prev_channel)
        self.B_prev.clicked.connect(self.prev_window)
        self.B_next.clicked.connect(self.next_window)
        self.a_shortcut.activated.connect(self.prev_window)
        self.d_shortcut.activated.connect(self.next_window)
        self.left_shortcut.activated.connect(self.prev_window)
        self.right_shortcut.activated.connect(self.next_window)
    
    def disconnect_navigation_buttons_shortcuts(self):
        """Disallow navigation between windows and channels.
        This is useful when we want to draw masks."""
        self.B_next_chan.clicked.disconnect(self.next_channel)
        self.B_prev_chan.clicked.disconnect(self.prev_channel)
        self.w_shortcut.activated.disconnect(self.next_channel)
        self.s_shortcut.activated.disconnect(self.prev_channel)
        self.down_shortcut.activated.disconnect(self.next_channel)
        self.up_shortcut.activated.disconnect(self.prev_channel)
        self.B_prev.clicked.disconnect(self.prev_window)
        self.B_next.clicked.disconnect(self.next_window)
        self.a_shortcut.activated.disconnect(self.prev_window)
        self.d_shortcut.activated.disconnect(self.next_window)
        self.left_shortcut.activated.disconnect(self.prev_window)
        self.right_shortcut.activated.disconnect(self.next_window)

    def next_channel(self):
        """Go to the next channel. Mask display is unchanged."""
        self.channel_idx = (self.channel_idx + 1) % self.n_channels
        self.update_window()
    
    def prev_channel(self):
        """Go to the previous channel. Mask display is unchanged."""
        self.channel_idx = (self.channel_idx - 1) % self.n_channels
        self.update_window()
    
    def next_window(self):
        """Go to the next grid of images and update masks."""
        if self.masks_shown:
            self.remove_masks()
        if self.window_idx < self.n_windows:
            self.window_idx += 1
        self.update_window()
        if self.masks_shown:
            self.add_masks()

    def prev_window(self):
        """Go to the previous grid of images and update masks."""
        if self.masks_shown:
            self.remove_masks()
        if self.window_idx > 0:
            self.window_idx -= 1
        self.update_window()
        if self.masks_shown:
            self.add_masks()
    
    def toggle_masks(self):
        """Toggle masks on or off."""
        if self.masks_shown:
            self.remove_masks()
        else:
            self.add_masks()
        self.masks_shown = not self.masks_shown
    
    def toggle_editable(self):
        """Toggle between clickable (label-cycling mode) 
        and editable (point-dragging mode) masks."""
        self.edit_mode = not self.edit_mode
        for i, j in np.ndindex(self.image_views.shape):
            curr_idx = i * self.grid_shape[1] + j
            try:
                for ma in self.masks[self.curr_indices[curr_idx]]:
                    ma.toggle_editable()
            except IndexError:
                continue
        
        if not self.masks_shown:
            self.toggle_masks()
            
    def add_masks(self):
        """For current ImageViews, add all relevant masks."""
        for i, j in np.ndindex(self.image_views.shape):
            curr_idx = i * self.grid_shape[1] + j
            try:
                for ma in self.masks[self.curr_indices[curr_idx]]:
                    # Toggle if there's a mismatch between edit 
                    # mode and the mask's editability
                    if ma.clickable == self.edit_mode:
                        ma.toggle_editable()
                    ma.add_to_imv(self.image_views[i, j])
            except IndexError:
                continue
    
    def remove_masks(self):
        """For current ImageViews, remove all relevant masks."""
        for i, j in np.ndindex(self.image_views.shape):
            curr_idx = i * self.grid_shape[1] + j
            try:
                for ma in self.masks[self.curr_indices[curr_idx]]:
                    ma.remove_from_imv()
            except IndexError:
                continue

    def freestyle(self):
        """ Toggle freestyle mode, where the user can draw masks
        by clicking and dragging on the ImageViews."""
        # Start freestyle mode by enabling drawing on all ImageViews
        if not self.freestyle_mode:
            self.freestyle_mode = True
            self.disconnect_navigation_buttons_shortcuts()  # Prevent navigation
            self.draw_val = int(np.max(self.curr_images) + 2)
            kernel = np.array([[self.draw_val]])
            for i, j in np.ndindex(self.image_views.shape):
                self.image_views[i, j].imageItem.setDrawKernel(kernel=kernel, 
                mask=None, center=(0,0))
            self.B_freestyle.setText("Finish drawing (r)")

        # End freestyle mode by creating a mask from the drawn points
        else:
            self.freestyle_mode = False
            self.connect_navigation_buttons_shortcuts()
            view_ranges = np.ndarray(shape=self.image_views.shape, dtype=object)
            for i, j in np.ndindex(self.image_views.shape):
                self.image_views[i, j].imageItem.setDrawKernel(kernel=None,
                    mask=None, center=None)

                # Get current viewbox state so we can reset it after drawing
                view_ranges[i, j] = self.image_views[i, j].view.getState()
                
                # Get points drawn
                im = self.image_views[i, j].imageItem.image
                mask = (im == self.draw_val)

                try:
                    points = get_mask_points(mask)
                    enclosed_mask = create_enclosed_mask(points, im.shape)
                    ma = ClickableEditableLabeledMask(enclosed_mask,
                                                      opacity=self.opacity,
                                                      possible_labels=self.possible_labels,
                                                      colors=self.colors,
                                                      clickable=(not self.edit_mode))
                    curr_idx = i * self.grid_shape[1] + j
                    self.masks[self.curr_indices[curr_idx]].append(ma)
                except IndexError:
                    continue

            self.update_window(view_ranges=view_ranges)
            if not self.masks_shown:
                self.toggle_masks()
            self.add_masks()
            self.draw_val += 2
            self.B_freestyle.setText("Draw masks (r)")

    def update_window(self, view_ranges=None):
        """Update the current window."""
        # Make sure window index is valid
        if self.window_idx < 0:
            self.window_idx = 0
        elif self.window_idx >= self.n_windows:
            self.window_idx = self.n_windows - 1
        
        # Make sure channel index is valid
        self.channel_idx = self.channel_idx % self.n_channels

        # Set title
        self.window.setWindowTitle("Cells {} to {}, channel {}" \
                                   .format(self.window_idx*self.n_images_displayed+1, 
                                           (self.window_idx+1)*self.n_images_displayed, 
                                           os.path.basename(self.snaps_folders[self.channel_idx])))

        # Get new images
        self.curr_indices = np.arange(self.window_idx * self.n_images_displayed,
                                      (self.window_idx+1) * self.n_images_displayed)
        self.curr_indices = self.curr_indices[self.curr_indices < self.n_snaps]
        curr_paths = list(self.snaps_filepaths[self.channel_idx, self.curr_indices])
        self.curr_images = tifffile.imread(curr_paths)
        if len(self.curr_images.shape) < 3:
            self.curr_images = np.expand_dims(self.curr_images, axis=0)
        # Update image views
        for i, j in np.ndindex(self.image_views.shape):
            curr_idx = i * self.grid_shape[1] + j
            # Add a blank image and ROI box if we are at the end of the data
            if curr_idx >= len(self.curr_indices):
                self.image_views[i, j].setImage(np.zeros((10, 10)))
                self.roi_views[i, j].setData(x=[], y=[], pen='r')
                continue

            self.image_views[i, j].setImage(self.curr_images[curr_idx])
            
            # Add ROI as a red box
            if self.rois is not None:
                self.roi_views[i, j].setData(x=[self.rois[curr_idx, 0],
                                                self.rois[curr_idx, 0],
                                                self.rois[curr_idx, 2],
                                                self.rois[curr_idx, 2],
                                                self.rois[curr_idx, 0]],
                                             y=[self.rois[curr_idx, 1],
                                                self.rois[curr_idx, 3],
                                                self.rois[curr_idx, 3],
                                                self.rois[curr_idx, 1],
                                                self.rois[curr_idx, 1]],
                                                pen='r')
            else:
                self.roi_views[i, j].setData(x=[], y=[], pen='r')
            
            # Set ViewBox states if provided
            if view_ranges is not None:
                try:
                    self.image_views[i, j].view.setState(view_ranges[i, j])
                except:
                    pass

    def save_state(self):
        """Save all masks as an NPZ file, giving the user a stopping point."""
        arrays_to_save = {}
        for (i,), masks in np.ndenumerate(self.masks):
            # Stop if we've reached the end of the masks
            if i >= self.n_snaps:
                break
            if len(masks) < 1:
                continue
            # Save points and indices for each mask
            curr_file_idx = str(i + 1)
            point_sets = []
            labels = []
            for ma in masks:
                point_sets.append(ma.points)
                labels.append(ma.idx)
            arrays_to_save[curr_file_idx + "_mask"] = create_integer_mask(point_sets, masks[0].shape)
            arrays_to_save[curr_file_idx + "_labels"] = np.asarray(labels)
    
        # Save integer masks as npz file
        np.savez_compressed(os.path.join(self.path, "saved_masks.npz"), 
                            **arrays_to_save)
        print("Masks saved!")
    
    def load_state(self):
        """Load NPZ masks."""
        save_state = os.path.join(self.path, "saved_masks.npz")
        if not os.path.exists(save_state):
            print(f"No saved masks found at {save_state}.")
            return
        
        self.remove_masks()
        saved_state = np.load(save_state)
        self.masks = np.ndarray(shape=self.n_windows * self.n_images_displayed, 
                                dtype=object)
        for j in tqdm(range(self.n_snaps), "Loading save state"):
            try:
                mask = saved_state[str(j+1) + "_mask"]
                labels = saved_state[str(j+1) + "_labels"]
                ma_indices = np.unique(mask[mask > 0])
                self.masks[j] = [ClickableEditableLabeledMask(mask == v, 
                                                              opacity=self.opacity,
                                                              idx=labels[i],
                                                              possible_labels=self.possible_labels,
                                                              colors=self.colors) \
                                for i, v in enumerate(ma_indices)]
            except KeyError:
                self.masks[j] = []
        
        if self.masks_shown:
            self.add_masks()
    
    def apply_masks(self):
        """Save and apply masks."""
        def mask_summary_plot(self):
            """Make a summary plot of the masks. This code was adapted
            from quot: https://github.com/alecheckert/quot"""
            x_max = self.rois[i, 2] - self.rois[i, 0]
            y_max = self.rois[i, 3] - self.rois[i, 1]
            valid_trajs = trajs[trajs["error_flag"] == 0.0]
            Y, X = np.indices((y_max, x_max))
            YX = np.asarray([Y.ravel(), X.ravel()]).T
            
            # Generate an image where each pixel is assigned to a mask
            mask_im = np.zeros((y_max, x_max), dtype=np.int64)
            for j, points in enumerate(point_sets):
                path = Path(points, closed=True)
                mask_im[path.contains_points(YX).reshape((y_max, x_max))] = j+1

            # Generate localization density
            y_bins = np.arange(y_max+1)
            x_bins = np.arange(x_max+1)
            H, _, _ = np.histogram2d(valid_trajs['y'], valid_trajs['x'], bins=(y_bins, x_bins))
            H = ndimage.gaussian_filter(H, 3.0)

            # The set of points to use for the scatter plot
            L = np.asarray(valid_trajs[["y", "x", "mask_index"]])

            # Categorize each localization as either (1) assigned or (2) not assigned
            # to a mask
            inside = L[:,2] > 0
            outside = ~inside 

            # Localization density in the vicinity of each spot
            yx_int = L[:,:2].astype(np.int64)
            densities = H[yx_int[:,0], yx_int[:,1]]
            norm = Normalize(vmin=0, vmax=densities.max())

            # Make a 3-panel plot
            plt.close('all')
            _, axs = plt.subplots(2, 2, figsize=(8, 8))
            filename = os.path.join(self.path, 'snaps3', f"{i+1}.tif")
            im = ImageReader(filename).get_frame(0)
            axs[0, 0].imshow(im, cmap='gray')
            axs[0, 1].imshow(mask_im, cmap='gray')
            axs[1, 0].imshow(H, cmap='gray')
            axs[1, 1].scatter(
                L[inside, 1],
                y_max-L[inside, 0],
                c=densities[inside],
                cmap="Greens",
                norm=norm,
                s=20
            )
            axs[1, 1].scatter(
                L[outside, 1],
                y_max-L[outside, 0],
                cmap="Reds",
                c=densities[outside],
                norm=norm,
                s=20
            )
            axs[1, 1].set_xlim((0, x_max))
            axs[1, 1].set_ylim((0, y_max))
            axs[1, 1].set_aspect('equal')

            # Subplot labels
            axs[0, 0].set_title("Masked image")
            axs[0, 1].set_title("Mask definitions")
            axs[1, 0].set_title("Localization density")
            axs[1, 1].set_title("Trajs inside/outside")

            for ax in axs.flatten():
                ax.axis('off')

            plt.savefig(os.path.join(self.path, "mask_plots", f"{i+1}.png"), 
                        dpi=600, 
                        bbox_inches='tight')
            plt.close('all')
        
        def mask_summary_csv(self):
            """Make a CSV with vertices and summary statistics for each FOV."""
            mask_dfs = []
            valid_idx = pd.unique(trajs[trajs['mask_index'] > 0]['mask_index'])
            if len(valid_idx) < 1:
                return
            for j, ma_index in enumerate(valid_idx):
                # Save a CSV file with trajectories inside only that mask                
                trajs_in_mask = trajs[trajs['mask_index'] == ma_index]
                out_path = os.path.join(self.path, 
                                        'masked_trajs', 
                                        f"{i+1}_{ma_index}_trajs.csv")
                trajs_in_mask.to_csv(out_path, index=False)
                n_dets = trajs_in_mask.shape[0]
                n_trajs = trajs_in_mask['trajectory'].nunique()
                n_jumps = n_dets - n_trajs

                # Save a CSV file with the mask points. 
                # Remember we flipped the points to facilitate 
                # quot masking, so the column order is 'y' then 'x'.
                mask_df = pd.DataFrame(point_sets[j], columns=['y', 'x'])
                mask_df['mask_index'] = ma_index
                for chan in self.snaps_folders:
                    filename = os.path.join(chan, f"{i+1}.tif")
                    im = ImageReader(filename).get_frame(0)
                    mask = valid_masks[j].mask
                    area = np.sum(mask)
                    mask_df["area"] = area
                    mask_df["eccentricity"] = get_eccentricity(mask)
                    mask_df['n_dets'] = n_dets
                    mask_df['n_trajs'] = n_trajs
                    mask_df['n_jumps'] = n_jumps
                    masked = mask * im
                    mask_df[f"{os.path.basename(chan)}_mean_intensity"] = np.sum(masked) / area
                mask_dfs.append(mask_df)
            mask_df = pd.concat(mask_dfs)
            out = os.path.join(self.path, 
                               'mask_measurements', 
                               f"{i+1}_masks.csv")
            mask_df.to_csv(out, index=False)

        # Make a save state
        self.save_state()
            
        # Not implemented yet for masks outside ROI or if no ROIs are provided.
        if self.rois is None:
            print("No ROIs provided, cannot apply masks.")
            return

        # Make sure output folders exist
        output_folders = ['masked_trajs', 'mask_plots', 'mask_measurements']
        for folder in output_folders:
            if not os.path.exists(os.path.join(self.path, folder)):
                os.makedirs(os.path.join(self.path, folder))

        # Save masks
        for (i,), masks in tqdm(np.ndenumerate(self.masks), 
                                total=self.n_snaps):
            # Stop if we've reached the end of the masks
            if i >= self.n_snaps:
                break
            # Get trajs.csv file,  make sure it exists
            trajs_csv = os.path.join(self.path, 'tracking', f"{i+1}.csv")
            if not os.path.isfile(trajs_csv):
                print(f"{trajs_csv} not found, skipping...")
                continue
            # Get masks to apply, skip if none are not 'bad'
            valid_masks = [mask for mask in masks if mask.curr_label != 'bad']
            if len(valid_masks) < 1:
                print(f"No masks to apply for {trajs_csv}, skipping...")
                continue

            # Apply masks and save; points are flipped for col-major quot masking
            point_sets = [np.flip(mask.points - [self.rois[i, 0], self.rois[i, 1]], 
                                  axis=1) 
                          for mask in valid_masks]
            trajs = pd.read_csv(trajs_csv)
            trajs['mask_index'] = apply_masks(point_sets, 
                                              trajs, 
                                              mode='all_points')
            trajs.to_csv(os.path.splitext(trajs_csv)[0] + "_trajs.csv", 
                         index=False)
            # Make a summary plot if requested
            if self.save_mask_png:
                mask_summary_plot(self)
            
            mask_summary_csv(self)


if __name__ == '__main__':
    print(f"qtpicker (v{VERSION}) is a GUI for selecting and editing masks.")
    # Get path to automation output folder
    path = str(input("Drag automation output folder: ").strip())
    if not os.path.isdir(path) and os.name == 'posix':
        path = path.replace("\\ ", " ")

    # If path is invalid, browse sample data
    if not os.path.isdir(path):
        print(f"{path} is not a valid path; browsing sample data instead...")
        path = os.path.join(os.path.dirname(__file__), "sample_data")

    # Run the GUI
    app = QApplication()
    window = ImageGrid(path,                            # automation output
                       shape=(2, 4),                    # shape of image grid to display
                       save_mask_png=True,              # save masking summary plots
                       roi_masks_only=True,             # only show masks within ROI
                       possible_labels=['good', 'bad'], # possible labels for masks
                       colors=['g', 'r'],               # colors corresponding to labels
                       min_mask_area=1000)              # reject masks smaller than this area
    app.exec()