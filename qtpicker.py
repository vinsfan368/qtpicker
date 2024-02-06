import sys

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
pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
from pyqtgraph import ImageView, ImageItem, PolyLineROI, mkColor, PlotDataItem
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 

# Image tools
from quot.read import ImageReader
from scipy import ndimage

# Masking tools
from quot.helper import get_edges, get_ordered_mask_points
from quot.gui.masker import apply_masks
from matplotlib.path import Path 

# Progress bar
from tqdm import tqdm

# Pickle
import pickle


def create_enclosed_mask(points, shape):
    """
    Create an enclosed mask from a list of points.

    Parameters:
    points: List of points. Each point is a tuple of (x, y).
    shape: Shape of the mask. A tuple of (height, width).

    Returns:
    An enclosed mask with the same shape. 
    The mask has a value of 1 inside the 
    enclosed area and 0 elsewhere.
    """
    y, x = np.indices(shape)
    coordinates = np.column_stack((x.flatten(), y.flatten()))
    mask = Path(points).contains_points(coordinates).reshape(shape)
    return mask

def get_mask_points(mask, max_points=25):
    """Wrapper for quot.helper.get_ordered_mask_points, executes
    this function and flips the points to match row-major order."""
    return np.flip(get_ordered_mask_points(mask, max_points=max_points), 
                   axis=1)


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
                 imv: ImageView=None, 
                 possible_labels: list[str]=['good', 'bad'],
                 colors: list[str]=['g', 'r'],
                 clickable: bool=True, 
                 opacity: float=0.15):
        
        self.mask = mask
        self.shape = mask.shape
        self.imv = imv
        self.opacity = opacity
        self.clickable = clickable
        self.points = get_mask_points(get_edges(mask))

        self.idx = 0
        self.possible_labels = possible_labels
        self.colors = colors
        self.curr_label = self.possible_labels[self.idx]
        self.curr_color = self.colors[self.idx]

        self.clickable_mask = ClickableMask(self)
        self.editable_mask = EditableMask(self)

    def add_to_imv(self, imv=None):
        """Add the mask to the image view, 
        overwriting self.imv if provided."""
        if imv is not None:
            self.imv = imv
        if self.imv is not None:
            if self.clickable:
                self.imv.addItem(self.clickable_mask)
            else:
                self.imv.addItem(self.editable_mask)
    
    def remove_from_imv(self, imv=None):
        """Remove the mask from the image view, 
        overwriting self.imv if provided."""
        if imv is not None:
            self.imv = imv
        if self.imv is not None:
            if self.clickable:
                self.imv.removeItem(self.clickable_mask)
            else:
                self.imv.removeItem(self.editable_mask)
    
    def toggle_editable(self):
        """Change between a clickable and an editable mask."""
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
        self.save_mask_png = save_mask_png
        self.show_hists = show_image_histogram
        self.roi_masks_only = roi_masks_only
        self.opacity = mask_opacity
        self.min_mask_area = min_mask_area
        self.possible_labels = possible_labels
        self.colors = colors
        self.freestyle_mode = False
        self.editable = False

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
            self.rois = np.loadtxt(os.path.join(self.path, "rois.txt"), 
                                   delimiter=',', 
                                   dtype=int)
            pad = self.sorted_data.shape[0] - self.rois.shape[0]
            self.rois = np.pad(self.rois, ((0, pad), (0, 0)))
        else:
            self.rois = None

        # Loop over snap images, snap folders
        for i in tqdm(range(self.n_snaps), "Loading images and masks"):
            for j, folder in enumerate(self.snaps_folders):
                # Load image into array
                filename = os.path.join(folder, f"{i+1}.tif")
                if os.path.exists(filename):
                    self.sorted_data[i, j] = ImageReader(filename).get_frame(0)

            # Get mask corresponding to this image and read in
            mask_filename = os.path.join(self.path, "masks", f"{i+1}.csv")
            if os.path.exists(mask_filename):
                # Load mask from CSV
                mask = np.loadtxt(mask_filename, delimiter=',')

                # Only look at masks inside ROI if specified
                if self.roi_masks_only and self.rois is not None:
                    # Get ROI for this image
                    roi = self.rois[i]
                    # Crop mask to ROI
                    crop = mask[roi[1]:roi[3], roi[0]:roi[2]]
                    # Get only unique indices within the ROI
                    idx, counts = np.unique(crop[crop > 0], return_counts=True)
                else:
                    # Else get all unique indices in the mask
                    idx, counts = np.unique(mask[mask > 0], return_counts=True)
                
                ma_indices = idx[counts > self.min_mask_area]
                    
                # Create a mask for each non-zero value
                self.masks[i] = [ClickableEditableLabeledMask(mask == v, 
                                                              opacity=self.opacity,
                                                              possible_labels=self.possible_labels,
                                                              colors=self.colors) \
                                 for v in ma_indices]

        # Reshape everything to (n_windows, grid_shape[0], grid_shape[1], ...)
        self.sorted_data = np.reshape(self.sorted_data, 
                                      newshape=(self.n_windows, 
                                                self.grid_shape[0], 
                                                self.grid_shape[1],
                                                len(self.snaps_folders),
                                                self.image_shape[1], 
                                                self.image_shape[2]))
        self.draw_val = np.max(self.sorted_data) + 2
        self.masks = np.reshape(self.masks, (self.sorted_data.shape[:3]))
        if self.rois is not None:
            self.rois = np.reshape(self.rois, (*self.sorted_data.shape[:3], 4))
    
    def init_UI(self):
        """Initialize the user interface."""
        self.window = QWidget()
        layout = QGridLayout(self.window)
        
        self.window_idx = 0
        self.channel_idx = 0
        self.masks_shown = True
        self.n_channels = len(self.snaps_folders)
        self.image_views = np.zeros((self.grid_shape[0], self.grid_shape[1]), 
                                    dtype=object)
        self.roi_views = np.zeros((self.grid_shape[0], self.grid_shape[1]),
                                    dtype=object)
        for i, j in np.ndindex(self.image_views.shape):
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

        # Add buttons to go through channels
        self.B_next_chan = QPushButton("Next channel (w, ↑)", self.window)
        self.B_prev_chan = QPushButton("Previous channel (s, ↓)", self.window)
        self.B_next_chan.clicked.connect(self.next_channel)
        self.B_prev_chan.clicked.connect(self.prev_channel)
        layout.addWidget(self.B_next_chan, i+1, 0)
        layout.addWidget(self.B_prev_chan, i+1, 1)

        # w shortcut to go to next channel, s to go to previous
        self.w_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_W), self.window)
        self.s_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_S), self.window)
        self.down_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Down), self.window)
        self.up_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Up), self.window)
        self.w_shortcut.activated.connect(self.next_channel)
        self.s_shortcut.activated.connect(self.prev_channel)
        self.down_shortcut.activated.connect(self.next_channel)
        self.up_shortcut.activated.connect(self.prev_channel)
        
        # Add buttons to advance windows
        self.B_prev = QPushButton("Previous window (a, ←)", self.window)
        self.B_next = QPushButton("Next window (d, →)", self.window)
        self.B_prev.clicked.connect(self.prev_window)
        self.B_next.clicked.connect(self.next_window)
        layout.addWidget(self.B_prev, i+1, 2)
        layout.addWidget(self.B_next, i+1, 3)

        # a and d key shortcuts to advance windows
        self.a_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_A), self.window)
        self.d_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_D), self.window)
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.window)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.window)
        self.a_shortcut.activated.connect(self.prev_window)
        self.d_shortcut.activated.connect(self.next_window)
        self.left_shortcut.activated.connect(self.prev_window)
        self.right_shortcut.activated.connect(self.next_window)

        # Add a button to toggle editable masks
        self.B_toggle_editable = QPushButton("Toggle editable (e)", self.window)
        self.B_toggle_editable.clicked.connect(self.toggle_editable)
        layout.addWidget(self.B_toggle_editable, i+2, 0)

        # e key shortcut to toggle editable masks
        self.e_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_E), self.window)
        self.e_shortcut.activated.connect(self.toggle_editable)

        # Add button to toggle showing masks
        self.B_toggle_masks = QPushButton("Toggle masks (q)", self.window)
        self.B_toggle_masks.clicked.connect(self.toggle_masks)
        layout.addWidget(self.B_toggle_masks, i+2, 1)

        # q key shortcut to toggle masks
        self.q_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Q), self.window)
        self.q_shortcut.activated.connect(self.toggle_masks)

        # Add a button to freestyle draw masks
        self.B_freestyle = QPushButton("Draw masks (r)", self.window)
        self.B_freestyle.clicked.connect(self.freestyle)
        layout.addWidget(self.B_freestyle, i+2, 2)

        # r shortcut to draw masks
        self.r_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_R), self.window)
        self.r_shortcut.activated.connect(self.freestyle)

        # Add a button to finish and apply masks
        self.B_apply_masks = QPushButton("Apply masks", self.window)
        self.B_apply_masks.clicked.connect(self.finish)
        layout.addWidget(self.B_apply_masks, i+2, 3)

        # Populate image views
        self.update_window()
        self.add_masks()

        # Resize main window
        self.window.resize(1280, 720)
        self.window.show()
    
    def next_channel(self):
        self.channel_idx = (self.channel_idx + 1) % self.n_channels
        self.update_window()
    
    def prev_channel(self):
        self.channel_idx = (self.channel_idx - 1) % self.n_channels
        self.update_window()
    
    def next_window(self):
        """Go to the next grid of images."""
        if self.masks_shown:
            self.remove_masks()
        if self.window_idx < self.n_windows:
            self.window_idx += 1
        self.update_window()
        if self.masks_shown:
            self.add_masks()

    def prev_window(self):
        """Go to the previous grid of images."""
        if self.masks_shown:
            self.remove_masks()
        if self.window_idx > 0:
            self.window_idx -= 1
        self.update_window()
        if self.masks_shown:
            self.add_masks()
    
    def toggle_masks(self):
        """Toggle masks on and off."""
        if self.masks_shown:
            self.remove_masks()
        else:
            self.add_masks()
        self.masks_shown = not self.masks_shown
    
    def toggle_editable(self):
        self.editable = not self.editable
        if not self.masks_shown:
            self.toggle_masks()
        for i, j in np.ndindex(self.image_views.shape):
            for mask_item in self.masks[self.window_idx, i, j]:
                mask_item.toggle_editable()
            
    def add_masks(self):
        """Add the masks to their respective ImageViews."""
        for i, j in np.ndindex(self.image_views.shape):
            for mask_item in self.masks[self.window_idx, i, j]:
                mask_item.add_to_imv(self.image_views[i, j])
    
    def remove_masks(self):
        """Remove masks from their respective ImageViews."""
        for i, j in np.ndindex(self.image_views.shape):
            for mask_item in self.masks[self.window_idx, i, j]:
                mask_item.remove_from_imv()
    
    def freestyle(self):
        #im = self.sorted_data[self.window_idx, 0, 0, self.channel_idx].copy()
        # Start freestyle mode by enabling drawing on all ImageViews
        if not self.freestyle_mode:
            self.freestyle_mode = True
            kernel = np.array([[self.draw_val]])
            for i, j in np.ndindex(self.image_views.shape):
                self.image_views[i, j].imageItem.setDrawKernel(kernel=kernel, 
                mask=None, center=(0,0))
            self.B_freestyle.setText("Finish drawing (r)")
        # End freestyle mode by creating a mask from the drawn points
        else:
            self.freestyle_mode = False
            for i, j in np.ndindex(self.image_views.shape):
                im = self.sorted_data[self.window_idx, i, j, self.channel_idx].copy() 
                self.image_views[i, j].imageItem.setDrawKernel(kernel=None,
                    mask=None, center=None)
                mask = (im == self.draw_val)
                try:
                    # Get points, initialize a new mask and add to list
                    free_points = get_mask_points(mask)
                    enclosed_mask = create_enclosed_mask(free_points, im.shape)
                    ma = ClickableEditableLabeledMask(enclosed_mask,
                                                      opacity=self.opacity,
                                                      possible_labels=self.possible_labels,
                                                      colors=self.colors,
                                                      clickable=(not self.editable))
        
                    self.masks[self.window_idx, i, j].append(ma)
                # If there are no points in an ImageItem, skip
                except IndexError:
                    continue
            
            self.add_masks()
            self.draw_val += 2
            self.B_freestyle.setText("Draw masks (r)")

    def update_window(self):
        """Update the current window."""
        # Make sure window index is valid
        if self.window_idx < 0:
            self.window_idx = 0
        elif self.window_idx >= self.n_windows:
            self.window_idx = self.n_windows - 1
        
        # Make sure channel index is valid
        self.channel_idx = self.channel_idx % self.n_channels
        self.window.setWindowTitle("Cells {} to {}, channel {}"\
                                   .format(self.window_idx*self.grid_shape[0]*self.grid_shape[1]+1, 
                                           (self.window_idx+1)*self.grid_shape[0]*self.grid_shape[1], 
                                           os.path.basename(self.snaps_folders[self.channel_idx])))
        for i, j in np.ndindex(self.image_views.shape):
            self.image_views[i, j].setImage(self.sorted_data[self.window_idx, 
                                                             i, 
                                                             j, 
                                                             self.channel_idx])
            
            # Add ROI as a red box
            if self.rois is not None:
                self.roi_views[i, j].setData(x=[self.rois[self.window_idx, i, j, 0],
                                                self.rois[self.window_idx, i, j, 0],
                                                self.rois[self.window_idx, i, j, 2],
                                                self.rois[self.window_idx, i, j, 2],
                                                self.rois[self.window_idx, i, j, 0]],
                                             y=[self.rois[self.window_idx, i, j, 1],
                                                self.rois[self.window_idx, i, j, 3],
                                                self.rois[self.window_idx, i, j, 3],
                                                self.rois[self.window_idx, i, j, 1],
                                                self.rois[self.window_idx, i, j, 1]],
                                                pen='r')
            else:
                self.roi_views[i, j].setData(x=[], y=[], pen='r')
        
    def mask_summary_plot(self):
        """Make a summary plot of the masks. This code was adapted
        from quot: https://github.com/alecheckert/quot"""
        pass
    
    def finish(self):
        """Save and apply masks."""
        # Not implemented yet for masks outside ROI or if no ROIs are provided.
        if self.rois is None:
            pass

        # Make sure output folders exist
        output_folders = ['masked_trajs', 'mask_plots', 'mask_measurements']
        for folder in output_folders:
            if not os.path.exists(os.path.join(self.path, folder)):
                os.makedirs(os.path.join(self.path, folder))

        masks_flat = self.masks.flatten()    
        rois = np.reshape(self.rois, (*masks_flat.shape, 4))
        # Save masks
        for (i,), masks in tqdm(np.ndenumerate(masks_flat), 
                                total=masks_flat.shape[0]):
            # Get trajs.csv file,  make sure it exists
            trajs_csv = os.path.join(self.path, 'tracking', f"{i+1}.csv")
            if not os.path.isfile(trajs_csv):
                print(f"{trajs_csv} not found, skipping...")
                continue
            # Get masks to apply, skip if none are good
            valid_masks = [mask for mask in masks if mask.curr_label != 'bad']
            if len(valid_masks) < 1:
                print(f"No masks to apply for {trajs_csv}, skipping...")
                continue
            # Apply masks and save; points need 
            # to be flipped to match row-major order
            point_sets = [np.flip(mask.points - [rois[i, 0], rois[i, 1]], axis=1) 
                          for mask in valid_masks]
            trajs = pd.read_csv(trajs_csv)
            trajs['mask_index'] = apply_masks(point_sets, 
                                              trajs, 
                                              mode='all_points')
            out_path = os.path.splitext(trajs_csv)[0] + "_trajs.csv"
            trajs.to_csv(out_path, index=False)
            if self.save_mask_png:
                x_max = rois[i, 2] - rois[i, 0]
                y_max = rois[i, 3] - rois[i, 1]
                trajs = trajs[trajs["error_flag"] == 0.0]
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
                H, _, _ = np.histogram2d(trajs['y'], trajs['x'], bins=(y_bins, x_bins))
                H = ndimage.gaussian_filter(H, 5.0)

                # The set of points to use for the scatter plot
                L = np.asarray(trajs[["y", "x", "mask_index"]])

                # Categorize each localization as either (1) assigned or (2) not assigned
                # to a mask
                inside = L[:,2] > 0
                outside = ~inside 

                # Localization density in the vicinity of each spot
                yx_int = L[:,:2].astype(np.int64)
                densities = H[yx_int[:,0], yx_int[:,1]]
                norm = Normalize(vmin=0, vmax=densities.max())

                # Make the 3-panel plot
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
            
            mask_dfs = []
            valid_idx = pd.unique(trajs[trajs['mask_index'] > 0]['mask_index'])
            if len(valid_idx) < 1:
                continue
            for j, ma_index in enumerate(valid_idx):
                # Save a CSV file with trajectories inside only that mask                
                trajs_in_mask = trajs[trajs['mask_index'] == ma_index]
                out = os.path.join(self.path, 
                                   'masked_trajs', 
                                   f"{i+1}_{ma_index}_trajs.csv")
                trajs_in_mask.to_csv(out, index=False)

                # Save a CSV file with the mask points
                mask_df = pd.DataFrame(point_sets[j], columns=['x', 'y'])
                mask_df['mask_index'] = ma_index
                for chan in self.snaps_folders:
                    filename = os.path.join(chan, f"{i+1}.tif")
                    im = ImageReader(filename).get_frame(0)
                    mask = valid_masks[j].mask
                    area = np.sum(mask)
                    mask_df["area"] = area
                    masked = mask * im
                    mask_df[f"{os.path.basename(chan)}_mean_intensity"] = np.sum(masked) / area
                mask_dfs.append(mask_df)
            mask_df = pd.concat(mask_dfs)
            out = os.path.join(self.path, 
                               'mask_measurements', 
                               f"{i+1}_masks.csv")
            mask_df.to_csv(out, index=False)

        # Save ImageGrid as pickle
        #with open(os.path.join(self.path, 'ImageGrid.pkl'), 'wb') as fh:
        #    pickle.dump(self, fh)


if __name__ == '__main__':
    path = str(input("Drag automation output folder: ").strip())
    if not os.path.isdir(path) and os.name == 'posix':
        path = path.replace("\\ ", " ")
    if not os.path.isdir(path):
        print(f"{path} is not a valid path; browsing sample data instead...")
        path = os.path.join(os.path.dirname(__file__), "sample_data")

    app = QApplication(sys.argv)
    window = ImageGrid(path, 
                       shape=(2, 4),
                       save_mask_png=True, 
                       roi_masks_only=True,
                       possible_labels=['good', 'bad'],
                       colors=['g', 'r'], 
                       min_mask_area=1000)
    app.exec()