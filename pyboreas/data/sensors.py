import os.path as osp
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

from pyboreas.data.pointcloud import PointCloud
from pyboreas.utils.utils import get_transform, yawPitchRollToRot, get_time_from_filename, load_lidar
from pyboreas.utils.utils import get_gt_data_for_frame, get_closest_index, get_inverse_tf
from pyboreas.utils.radar import load_radar, radar_polar_to_cartesian
from pyboreas.vis.vis_utils import vis_lidar, vis_camera, vis_radar
from pyboreas.data.bounding_boxes import BoundingBoxes


class Sensor:
    def __init__(self, path):
        self.path = path
        self.labelFolder = None
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
        """Initializes pose variables with ground truth applanix data
        Args:
            data (list): A list of floats corresponding to the line from the sensor_pose.csv file
                with the matching timestamp
        """
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

    def get_bounding_boxes(self, seqLabelFiles=[], seqLabelTimes=[], seqLabelPoses=[]):
        self.bbs = BoundingBoxes()
        labelPath = osp.join(self.seq_root, self.labelFolder, self.frame + '.txt')
        if osp.exists(labelPath):
            self.bbs.load_from_file(labelPath)
        else:
            if len(seqLabelFiles) == 0 or len(seqLabelTimes) == 0 or len(seqLabelPoses) == 0:
                return None
            idx = get_closest_index(self.timestamp, seqLabelTimes)
            if idx == 0 or idx == len(seqLabelTimes) - 1:
                self.bbs.load_from_file(seqLabelFiles[idx])
                T_enu_lidar = seqLabelPoses[idx]
                T = np.matmul(get_inverse_tf(self.pose), T_enu_lidar)
                self.bbs.transform(T)
            else:
                self.bbs.interpolate(idx, self.timestamp, self.pose, seqLabelFiles, seqLabelTimes, seqLabelPoses)
        return self.bbs


class Lidar(Sensor, PointCloud):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.points = None
        self.bbs = None

    def load_data(self):
        self.points = load_lidar(self.path)
        return self.points

    def visualize(self, **kwargs):
        return vis_lidar(self, **kwargs)

    def unload_data(self):
        self.points = None

    def has_bbs(self):
        labelPath = osp.join(self.seq_root, self.labelFolder, self.frame + '.txt')
        return osp.exists(labelPath)


class Camera(Sensor):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.img = None

    def load_data(self):
        img = cv2.imread(self.path)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.img

    def visualize(self, **kwargs):
        return vis_camera(self, **kwargs)

    def unload_data(self):
        self.img = None


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
        # Loads polar radar data, timestamps, azimuths, and resolution value
        # Additionally, loads a pre-computed cartesian radar image and binary mask if they exist.
        self.timestamps, self.azimuths, _, self.polar, self.resolution = load_radar(self.path)
        cart_path = osp.join(self.sensor_root, 'cart', self.frame + '.png')
        if osp.exists(cart_path):
            self.cartesian = cv2.imread(cart_path, cv2.IMREAD_GRAYSCALE)
        mask_path = osp.join(self.sensor_root, 'mask', self.frame + '.png')
        if osp.exists(mask_path):
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return self.timestamps, self.azimuths, self.polar

    def unload_data(self):
        self.timestamps = None
        self.azimuths = None
        self.polar = None
        self.cartesian = None
        self.mask = None

    def polar_to_cart(self, cart_resolution, cart_pixel_width, polar=None, in_place=True):
        """Converts a polar scan from polar to Cartesian format
        Args:
            cart_resolution (float): resolution of the output Cartesian image in (m / pixel)
            cart_pixel_width (int): width of the output Cartesian image in pixels
            polar (np.ndarray): if supplied, this function will use this input and not self.polar.
            in_place (bool): if True, self.cartesian is updated.
        """
        if polar is None:
            polar = self.polar
        cartesian = radar_polar_to_cartesian(self.azimuths, polar, self.resolution,
                                             cart_resolution, cart_pixel_width)
        if in_place:
            self.cartesian = cartesian
        return cartesian

    def visualize(self, **kwargs):
        return vis_radar(self, **kwargs)

