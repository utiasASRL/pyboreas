import numpy as np
import cv2

from pointcloud import PointCloud
from utils.utils import get_transform, load_lidar
from utils.radar import load_radar, radar_polar_to_cartesian


class Sensor:
    def __init__(self, path):
        self.path = path
        self.frame = path.split('/')[-1]
        self.sensType = path.split('/')[-2]
        self.seqID = path.split('/')[-3]
        self.root = '/'.join(path.split('/')[:-2] + [''])
        self.pose = None
        self.velocity = None
        self.body_rate = None
        self.timestamp = None

    def get_pose(self):
        if self.pose is not None:
            return self.pose
        posepath = self.root + 'applanix/' + self.sensType + '_poses.csv'
        with open(posepath, 'r') as f:
            f.readline()  # header
            for line in f:
                if line.split(',')[0] == self.frame:
                    gt = [float(x) for x in line.split(',')]
                    self.pose = get_transform(gt)
                    wbar = np.array([gt[12], gt[11], gt[10]]).reshape(3, 1)
                    wbar = np.matmul(self.pose[:3, :3], wbar)
                    self.velocity = np.array([gt[4], gt[5], gt[6], wbar[0], wbar[1], wbar[2]]).reshape(6, 1)
                    vbar = np.array([gt[4], gt[5], gt[6]]).reshape(3, 1)
                    vbar = np.matmul(self.pose[:3, :3].T, vbar)
                    self.body_rate = np.array([vbar[0], vbar[1], vbar[2], gt[12], gt[11], gt[10]])
                    self.timestamp = gt[1]
                    return self.pose

    def get_velocity(self):
        if self.velocity is None:
            self.get_pose()
        return self.velocity

    def get_body_rate(self):
        if self.body_rate is None:
            self.get_pose()
        return self.body_rate

    def get_timestamp(self):
        if self.timestamp is None:
            self.get_pose()
        return self.timestamp


class Lidar(Sensor, PointCloud):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.points = load_lidar(path)
        self.timestamp = None


# TODO: get_bounding_boxes()
# TODO: get_semantics()
# TODO: visualize(int: projection, bool: use_boxes)

class Camera(Sensor):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.img = cv2.imread(path)


# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: get_semantics() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes, Lidar: points (optional_arg))

class Radar(Sensor):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.timestamps, self.azimuths, _, self.polar = load_radar(path)

    def get_cartesian(self, radar_resolution, cart_resolution, cart_pixel_width):
        return radar_polar_to_cartesian(self.azimuths, self.polar, radar_resolution,
                                        cart_resolution)

# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
# TODO: visualize(bool: use_boxes)