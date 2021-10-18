import os.path as osp
import numpy as np
import cv2
from pathlib import Path

from pyboreas.data.pointcloud import PointCloud
from pyboreas.utils.utils import get_transform, yawPitchRollToRot, get_time_from_filename, load_lidar
from pyboreas.utils.utils import get_gt_data_for_frame
from pyboreas.utils.radar import load_radar, radar_polar_to_cartesian

class Sensor:
    def __init__(self, path):
        self.path = path
        p = Path(path)
        self.frame = p.stem
        self.sensType = p.parts[-2]
        self.seqID = p.parts[-3]
        self.seq_root = str(Path(*p.parts[:-2]))
        self.sensor_root = osp.join(self.seq_root, self.sensType)
        self.pose = np.identity(4, dtype=np.float64)  # T_enu_sensor
        self.velocity = np.zeros((6, 1))   # 6 x 1 velocity in ENU frame [v_se_in_e; w_se_in_e] 
        self.body_rate = np.zeros((6, 1))  # 6 x 1 velocity in sensor frame [v_se_in_s; w_se_in_s]
        self.timestamp = get_time_from_filename(self.frame)

    def init_pose(self, data=None):
        if data is not None:
            gt = [float(x) for x in data]
        else:
            gt = get_gt_data_for_frame(self.seq_root, self.sensType, self.frame)
        self.pose = get_transform(gt)
        wbar = np.array([gt[12], gt[11], gt[10]]).reshape(3, 1)
        wbar = np.matmul(self.pose[:3, :3], wbar).squeeze()
        self.velocity = np.array([gt[4], gt[5], gt[6], wbar[0], wbar[1], wbar[2]]).reshape(6, 1)
        vbar = np.array([gt[4], gt[5], gt[6]]).reshape(3, 1)
        vbar = np.matmul(self.pose[:3, :3].T, vbar).squeeze()
        self.body_rate = np.array([vbar[0], vbar[1], vbar[2], gt[12], gt[11], gt[10]]).reshape(6, 1)

class Lidar(Sensor, PointCloud):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.points = None

    def load_data(self):
        self.points = load_lidar(self.path)
        return self.points

# TODO: get_bounding_boxes()
# TODO: get_semantics()
# TODO: visualize(int: projection, bool: use_boxes)

class Camera(Sensor):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.img = None

    def load_data(self):
        self.img = cv2.imread(self.path)
        return self.img

# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: get_semantics() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes, Lidar: points (optional_arg))

class Radar(Sensor):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.resolution = 0.0596
        self.timestamps = None
        self.azimuths = None
        self.polar = None
        self.cartesian = None
        self.mask = None

    def load_data(self):
        self.timestamps, self.azimuths, _, self.polar, self.resolution = load_radar(self.path)
        cart_path = osp.join(self.sensor_root, 'cart', self.frame + '.png')
        if osp.exists(cart_path):
            self.cartesian = cv2.imread(cart_path, cv2.IMREAD_GRAYSCALE)
        mask_path = osp.join(self.sensor_root, 'mask', self.frame + '.png')
        if osp.exists(mask_path):
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return self.timestamps, self.azimuths, self.polar

    def get_cartesian(self, cart_resolution, cart_pixel_width, polar=None):
        if polar is None:
            polar = self.polar
        self.cartesian = radar_polar_to_cartesian(self.azimuths, polar, self.resolution,
                                        cart_resolution, cart_pixel_width)

# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes)