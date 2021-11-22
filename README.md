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
1. [Create an AWS account (optional)](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
3. Create a `root` folder to store the dataset, example: `/path/to/data/boreas/` Each sequence will then be a folder under `root`.
4. Use the AWS CLI to download either the entire dataset or only the desired sequences and sensors. Add `--no-sign-request` after each of the following commands if you're not going to use an AWS account. For example, the following command will download the entire Boreas dataset:

```bash
root=/path/to/data/boreas/
aws s3 sync s3://boreas $root
```

The following command will list all the top-level prefixes (sequences):

```bash
root=/path/to/data/boreas/
aws s3 ls s3://boreas
```

Alternatively, [boreas.utias.utoronto.ca (Work In Progress)](https://www.boreas.utias.utoronto.ca/#/download) can be used to browse through sequences so as to pick and choose what data to download. The website will then generate a list of AWS CLI commands that can be run as a bash script. These commands will look something like:

```bash
root=/path/to/data/boreas/
cd $root
aws s3 sync s3://boreas/boreas-2020-11-26-13-58 boreas-2020-11-26-13-58 --exclude "*" \
    --include "lidar/" --include "radar/" \
    --include "applanix/" --include "calib/"
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
# and groundtruth velocity information. Unless it's a part of the test set,
# in that case, ground truth poses will be missing. However we still provide IMU
# data (in the applanix frame) through the imu.csv files.
lidar_frame = bd.get_lidar(0)
t = lidar_frame.timestamp  # timestamp in seconds
T_enu_lidar = lidar_frame.pose  # 4x4 homogenous transform [R t; 0 0 0 1]
vbar = lidar_frame.velocity  # 6x1 vel in ENU frame [v_se_in_e; w_se_in_e] 
varpi = lidar_frame.body_rate  # 6x1 vel in sensor frame [v_se_in_s; w_se_in_s]
```
## Data Visualizer
We provide a tool for visualization of sequence frames. Currently, the visualizer supports BEV lidar visualization, BEV radar visualization, Perspective camera + lidar visualization, and 3D lidar point visualization.

```Python
from pyboreas import BoreasDataset
from pyboreas.vis.visualizer import BoreasVisualizer

root = '/path/to/data/boreas/'
bd = BoreasDataset(root)
seq = bd.sequences[0]

bv = BoreasVisualizer(seq)
bv.visualize(starting_frame_idx=0)
```
Running the above code will start a local web server that visualizes the selected sequence.
```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'pyboreas.vis.visualizer' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
```
Open a web browser and navigate to the provided ip (in this case 127.0.0.1:8050) to view the sequence visualization.
## Tutorials
Note that we provide a few simple tutorials for getting started with the Boreas dataset. Also note that we provide instructions for using this dataset using an AWS SageMaker instance, instructions at: pyboreas/tutorials/aws/README.md.

**NOTE:** ground truth poses have dtype=np.float64, but PyTorch defaults to float32. Avoid using implicit type conversion as this will result in significant quantization error. Implicit conversion is only safe when the translation values are small, such as a pose with respect to a sensor frame or with respect to a starting position, but NOT with respect to ENU (very large).

TODO:
- Tutorials (pose interp)
- Ground plane removal
- Pointcloud voxelization
- 3D Bounding boxes
