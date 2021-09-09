from os import path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from pointcloud import PointCloud
from utils.utils import get_transform
from utils.radar import load_radar, radar_polar_to_cartesian


class Sensor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.frame = path.splitext(data_path.split('/')[-1])[0]
        self.sensType = data_path.split('/')[-2]
        self.seqID = data_path.split('/')[-3]
        self.pose = None
        self.velocity = None
        self.body_rate = None
        self.gps_ts = None
        self.ros_ts = None

    def init_pose(self, data):
        self.transform = get_transform([float(x) for x in data])
        self.position = np.asarray(data[2:5], dtype=np.float32)
        self.velocity = np.asarray(data[5:8], dtype=np.float32)
        self.rotation = np.asarray(data[8:11], dtype=np.float32)
        self.gps_ts = float(data[1])
        self.ros_ts = int(data[0])

    def get_pose(self):
        if self.pose is not None:
            return self.pose
        else:
            raise KeyError('Pose is uninitialized')

    def get_velocity(self):
        if self.velocity is None:
            self.get_pose()
        return self.velocity

    def get_body_rate(self):
        if self.body_rate is None:
            self.get_pose()
        return self.body_rate

    def get_timestamp(self):
        return self.ros_ts  # Currently set to ROS TS, maybe change once we get better time standard


class Lidar(Sensor, PointCloud):
    def __init__(self, data_path):
        Sensor.__init__(self, data_path)

    def get_C_v_enu(self):
        return R.from_euler('xyz', self.rotation)

    def load_points(self, dim=6):
        scan = np.fromfile(self.data_path, dtype=np.float32)
        points = scan.reshape((-1, dim))[:, :dim]
        return points


# TODO: get_bounding_boxes()
# TODO: get_semantics()
# TODO: visualize(int: projection, bool: use_boxes)

class Camera(Sensor):
    def __init__(self, data_path):
        Sensor.__init__(self, data_path)

    def load_img(self):
        self.img = cv2.imread(self.data_path)


# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: get_semantics() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes, Lidar: points (optional_arg))

class Radar(Sensor):
    def __init__(self, data_path):
        Sensor.__init__(self, data_path)

    def load_scan(self):
        return load_radar(self.data_path)

    def get_cartesian(self, radar_resolution, cart_resolution, cart_pixel_width):
        return radar_polar_to_cartesian(self.azimuths, self.polar, radar_resolution,
                                        cart_resolution)

# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes)