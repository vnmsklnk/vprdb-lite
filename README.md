# VPR-DB-Lite
VPR-DB-Lite is a tool for creating optimally sized databases 
(containing the minimum number of frames covering the scene) 
for place recognition task from RGB-D data using dominating set algo.

## Datasets format
To use the tool, your data must be in a specific format.
* Color images in any format.
* Depth images corresponding to color images in 16-bit grayscale format.
* The trajectory containing one pose in each line in `timestamp tx ty tz qx qy qz qw` format.
* Camera intrinsics as a `.txt` file

Therefore, the structure of the dataset should look like this:
```
Example dataset
├── color
|   ├── 001.png
|   ├── 002.png
|   ├── ...
├── depth
|   ├── 001.pcd
|   ├── 002.pcd
|   ├── ...
├── intrinsics.txt
└── CameraTrajectory.txt
```
The number of color images, depth images and poses 
in the trajectory file must be the same.

## Usage
Please, check `pipeline.ipynb` with usage example.