# qtpicker
Cell/FOV QC UI for automated imaging, to be integrated into an all-encompassing UI.

## Dependencies
`quot` (https://github.com/alecheckert/quot) and its dependencies and scikit-image >= 0.24

## Using `qtpicker`
Clone the repository with `git clone https://github.com/vinsfan368/qtpicker.git`. In an environment with `quot` installed, run `qtpicker` with `python qtpicker.py`, then follow the command-line prompts.

`qtpicker` currently expects a very rigid directory and file naming scheme, where `n` is the number of fields of view to analyze:
```
── automation_output_folder
    ├── "snaps2" [a FOV that includes all detected particles]
    |   ├── "1.tif"
    |   ├── "2.tif"
    |   ├── ...
    |   └── n.tif
    ├── snaps2_another_channel [optional extra channel(s)]
    |   ├── "1.tif"
    |   ├── "2.tif"
    |   ├── ...
    |   └── n.tif 
    ├── "masks" [CSV masks, perhaps generated by some segmenting algorithm]
    |   ├── "1.csv" [CSV file where background is 0 and masks are positive numbers]
    |   ├── "2.csv"
    |   ├── ...
    |   └── n.csv
    ├── "tracking" [tracks generated by quot]
    |   ├── "1.csv"
    |   ├── "2.csv"
    |   ├── ...
    |   └── n.csv
    └── "rois.txt" [comma-separated vals of shape (n, 4), describing how the FOV in snaps2 is cropped] 
```

`qtpicker` outputs into the automation output folder:
```
── automation_output_folder
    ├── "masked_trajs" [trajectories in masks, one file per mask]
    |   ├── "1_1_trajs.csv" [first mask in the first FOV]
    |   ├── "1_2_trajs.csv"
    |   ├── "2_1_trajs.csv" 
    |   ├── ...
    |   └── n_j_trajs.csv [jth mask in the nth FOV]
    ├── "mask_measurements" [files containing mask vertices and fluorescence vals, one file per FOV]
    |   ├── "1_masks.csv"
    |   ├── "2_masks.csv"
    |   ├── ...
    |   └── n_masks.csv
    ├── "mask_plots" [PNG files summarizing masks applied to each FOV, one per FOV. Optional, outputted only if save_mask_png=True]
    |   ├── "1.png"
    |   ├── "2.png"
    |   ├── ...
    |   └── n.png
    └── "saved_masks.npz" [A save state of masks modified/drawn by the user, so that the user can pick up where s/he left off]
```

## Updating `qtpicker`
In your `qtpicker` directory, use `git pull` to fetch the latest commit.
