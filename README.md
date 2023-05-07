# Parallel_Suite2p
"""
# ImageProcessor Documentation
This Python code defines a class called `ImageProcessor` that is used to process and analyze TIFF images. The class is designed to handle mesoscopic data and multiplane images. The ImageProcessor class provides functionalities to extract ROI dimensions, load and save image data, perform parallel processing, and reconstruct images.
## Installation

    1. Install suite2p following instructions here: https://github.com/MouseLand/suite2p
    2. Clone or download current repo
   ### Usage
    3. Open anaconda prompt
    4. Navigate to the directory where this repo exists by typing "cd: path-to-the-repo", or add it to your path after opening jupyter notebook
    5. Activate the suite2p environment by typing "activate suite2p"
    6. Type "jupyter notebook" to lunch the jupyter server in your web-browser
    7. Go to the example notebook and run the cells
    8. After analyzing your data, you can follow the rest of the steps for closed-loop reactivation of functionally interesting neurons on this repo: (TODO ADD REPO)
    

## Dependencies

This code should be run from within the suite2p environment.
The following libraries are required (All of which are contained in the suite2p package):
- os
- gc
- time
- h5py
- numpy as np
- json
- pathlib.Path
- suite2p.run_s2p
- ScanImageTiffReader.ScanImageTiffReader
- multiprocessing
- matplotlib.pyplot as plt

## Class: ImageProcessor

The `ImageProcessor` class has the following methods:

### `__init__(self, tiff_folder_list=[], exp_list=[], x_bounds=[0,512], y_bounds=[0,512], out_path='out_path', channels=1, channelOI=0, singleChannel=1)`

The constructor initializes the ImageProcessor object with the given parameters.

- `tiff_folder_list`: A list of paths to TIFF folders.
- `exp_list`: A list of experiment names corresponding to the TIFF folders. If not provided, the names will be derived from the folder paths.
- `x_bounds`: A list containing the lower and upper bounds of the x-axis cropping window in pixels (default: [0, 512]).
- `y_bounds`: A list containing the lower and upper bounds of the y-axis cropping window in pixels (default: [0, 512]).
- `out_path`: The output path where the processed data will be saved (default: 'out_path').
- `channels`: The number of channels in the input images (default: 1).
- `channelOI`: The channel of interest to be processed (default: 0). 
- `singleChannel`: A flag to indicate if the h5 files should contain only the channel of interest (default: 1).

### `create_directory(self, path)`

This method creates a new directory at the specified path if it doesn't already exist.

- `path`: The path where the new directory will be created.

### `get_meta_info(self)`

This method retrieves mesoscopic data and metadata from the first TIFF image in the specified folder list. 

### `plotReconstruction(self)`

This method generates a plot of the reconstructed image from the processed data.

### `save_h5s(self)`

This method saves the sliced/croped image data from independent ROIS into individual files using an HDF5 (.h5) format in the specified output directory.

### `crop_and_save_h5(self, exp_name, tiff_folder)`

[[sub-method of 'save_h5s']] This method  crops, slices and saves the multiplane image data in the HDF5 (.h5) format.

- `exp_name`: The name of the experiment.
- `tiff_folder`: The path to the TIFF folder.

### `crop_and_save_h5_meso(self, exp_name, tiff_folder)`

[[sub-method of 'save_h5s']] This method crops, slices and saves the mesoscopic MROI image data in the HDF5 (.h5) format.

- `exp_name`: The name of the experiment.
- `tiff_folder`: The path to the TIFF folder.

### `load_tif(self, mov_path)`

This method loads a SCANIMAGE TIFF image from the specified path. Optionally, it takes only the channel of interest.

- `mov_path`: The path to the TIFF image file.

### `run_parallel_processing(self, ops)`

This method runs parallel processing of each independent .h5 files previously saved using Suite2p.

- `ops`: A dictionary containing Suite2p options.
"""
Parallelized suite2p for closed-loop all-optical experiments. Both mesoscopic and standard 2-photon scanning systems are supported
