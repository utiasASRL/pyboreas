# pyboreas
![Boreas](https://github.com/utiasASRL/pyboreas/blob/master/pyboreas/figs/pyboreas.png)

This devkit provides tools for working with the Boreas Dataset, an all-weather autonomous driving dataset which includes a 128-beam Velodyne Alpha-Prime lidar, a 5MP Blackfly camera, a 360 degree Navtech radar, and post-processed Applanix POS LV GNSS data. Our dataset currently suports benchmarking odometry. We are working towards providing an online benchmark for odometry, localization, and more. We plan to provide an HD map of each driven route. We are also in the process of acquiring 3D labels and hope to be able to provide a challenging object detection benchmark in the future.

Please note that our website is currently under construction. A live benchmark and a browser for downloading sequences will be available via the website soon.

## Installation

### Using pip
```
pip install asrl-pyboreas
```

### From source
```
git clone https://github.com/utiasASRL/pyboreas.git
pip install -e pyboreas
```

## Download Instructions
1. [Create an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
3. Create a `root` folder to store the dataset, example: `/path/to/data/boreas/` Each sequence will then be a folder under `root`.
4. Use the AWS CLI to download either the entire dataset or only the desired sequences and sensors. For example, the following command will download the entire Boreas dataset:

```bash
root=/path/to/data/boreas/
aws s3 sync s3://boreas $root
```

Alternatively, [our website (Work-In-Progress)](boreas.utias.utoronto.ca/#/download) can be used to browse through sequences so as to pick and choose what data to download. The website will then generate a list of AWS CLI commands that can be run as a bash script. These commands will look something like:

```bash
root=/path/to/data/boreas/
cd $root
aws s3 sync s3://boreas/boreas-2020-11-26-13-58 ./boreas-2020-11-26-13-58 --exclude "*" \
    --include "lidar/" --include "radar/" \
    --include "applanix/" --include "calib/"
```

The folder structure should end up looking like:
```
$ ls /path/to/data/boreas/
boreas-2020-11-26-13-58
boreas-2020-12-01-13-26
...
$ ls /path/to/data/boreas/boreas-2020-11-26-13-58
applanix calib camera lidar radar
```

## Example Usage

```Python
import numpy as np
from pyboreas import BoreasDataset

root = '/path/to/data/boreas/'
bd = BoreasDataset(root)

# Note: The Boreas dataset differs from others (KITTI) in that camera,
# lidar, and radar measurements are not synchronous. However, each
# sensor message has an accurate timestamp and pose instead.
# See our tutorials for how to work with multiple sensors.

# Loop through each frame in order (odometry)
for seq in bd.sequences:
    # Iterator examples:
    for camera_frame in seq.camera:
        img = camera_frame.img  # np.ndarray
        # do something
        camera_frame.unload_data() # Memory reqs will keep increasing without this
    for lidar_frame in seq.lidar:
        pts = lidar_frame.points  # np.ndarray (x,y,z,i,r,t)
        # do something
        lidar_frame.unload_data() # Memory reqs will keep increasing without this
    # Retrieve frames based on their index:
    N = len(seq.radar_frames)
    for i in range(N):
        radar_frame = seq.get_radar(i)
        # do something
        radar_frame.unload_data() # Memory reqs will keep increasing without this

# Iterator example:
cam_iter = bd.sequences[0].get_camera_iter()
cam0 = next(cam_iter)  # First camera frame
cam1 = next(cam_iter)  # Second camera frame

# Randomly access frames (deep learning, localization):
N = len(bd.lidar_frames)
indices = np.random.permutation(N)
for idx in indices:
    lidar_frame = bd.get_lidar(idx)
    # do something
    lidar_frame.unload_data() # Memory reqs will keep increasing without this

# Each sequence contains a calibration object:
calib = bd.sequences[0].calib
point_lidar = np.array([1, 0, 0, 1]).reshape(4, 1)
point_camera = np.matmul(calib.T_camera_lidar, point_lidar)

# Each sensor frame has a timestamp, groundtruth pose
# (4x4 homogeneous transform) wrt a global coordinate frame (ENU),
# and groundtruth velocity information.
lidar_frame = bd.get_lidar(0)
t = lidar_frame.timestamp  # timestamp in seconds
T_enu_lidar = lidar_frame.pose  # 4x4 homogenous transform [R t; 0 0 0 1]
vbar = lidar_frame.velocity  # 6x1 vel in ENU frame [v_se_in_e; w_se_in_e] 
varpi = lidar_frame.body_rate  # 6x1 vel in sensor frame [v_se_in_s; w_se_in_s]
```

## Tutorials
Note that we provide a few simple tutorials for getting started with the Boreas dataset. Also note that we provide instructions for using this dataset using an AWS SageMaker instance, instructions at: pyboreas/tutorials/aws/README.md.

TODO:
- Tutorials (pose interp)
- Convert readme pdf to markdown
- Ground plane removal
- Pointcloud voxelization
- 3D Bounding boxes
