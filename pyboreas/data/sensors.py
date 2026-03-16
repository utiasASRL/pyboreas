import os.path as osp
from pathlib import Path

import cv2
import numpy as np

from pyboreas.data.bounding_boxes import BoundingBoxes
from pyboreas.data.pointcloud import PointCloud
from pyboreas.utils.radar import load_radar, radar_polar_to_cartesian
from pyboreas.utils.utils import (
    get_closest_index,
    get_gt_data_for_frame,
    get_inverse_tf,
    get_time_from_filename,
    get_time_from_filename_microseconds,
    get_transform,
    load_lidar,
)
from pyboreas.vis.vis_utils import vis_camera, vis_lidar, vis_radar


class Sensor:
    def __init__(self, path):
        self.path = path
        self.labelFolder = "labels"
        p = Path(path)
        self.frame = p.stem
        self.sensType = None
        self.seqID = None
        self.seq_root = None
        self.sensor_root = None
        if len(p.parts) >= 2:
            self.sensType = p.parts[-2]
            self.seq_root = str(Path(*p.parts[:-2]))
            self.sensor_root = osp.join(self.seq_root, self.sensType)
        if len(p.parts) >= 3:
            self.seqID = p.parts[-3]
        self.pose = np.identity(4, dtype=np.float64)  # T_enu_sensor
        self.velocity = np.zeros(
            (6, 1)
        )  # 6 x 1 velocity in ENU frame [v_se_in_e; w_se_in_e]
        self.body_rate = np.zeros(
            (6, 1)
        )  # 6 x 1 velocity in sensor frame [v_se_in_s; w_se_in_s]
        try:
            self.timestamp = get_time_from_filename(self.frame)
            self.timestamp_micro = get_time_from_filename_microseconds(self.frame)
        except:
            self.timestamp = 0
            self.timestamp_micro = 0

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
        self.velocity = np.array(
            [gt[4], gt[5], gt[6], wbar[0], wbar[1], wbar[2]]
        ).reshape(6, 1)
        vbar = np.array([gt[4], gt[5], gt[6]]).reshape(3, 1)
        vbar = np.matmul(self.pose[:3, :3].T, vbar).squeeze()
        self.body_rate = np.array(
            [vbar[0], vbar[1], vbar[2], gt[12], gt[11], gt[10]]
        ).reshape(6, 1)

    def get_bounding_boxes(self, seqLabelFiles=[], seqLabelTimes=[], seqLabelPoses=[]):
        self.bbs = BoundingBoxes()
        labelPath = osp.join(self.seq_root, self.labelFolder, self.frame + ".txt")
        if osp.exists(labelPath):
            self.bbs.load_from_file(labelPath)
        else:
            if (
                len(seqLabelFiles) == 0
                or len(seqLabelTimes) == 0
                or len(seqLabelPoses) == 0
            ):
                return None
            idx = get_closest_index(self.timestamp, seqLabelTimes)
            if idx == 0 or idx == len(seqLabelTimes) - 1:
                self.bbs.load_from_file(seqLabelFiles[idx])
                T_enu_lidar = seqLabelPoses[idx]
                T = np.matmul(get_inverse_tf(self.pose), T_enu_lidar)
                self.bbs.transform(T)
            else:
                self.bbs.interpolate(
                    idx,
                    self.timestamp,
                    self.pose,
                    seqLabelFiles,
                    seqLabelTimes,
                    seqLabelPoses,
                )
        return self.bbs


class Lidar(Sensor, PointCloud):
    def __init__(self, path):
        Sensor.__init__(self, path)
        self.points = None
        self.bbs = None
        self._dim = 6

    def load_data(self):
        self.points = load_lidar(self.path, dim=self._dim)
        return self.points

    def visualize(self, **kwargs):
        return vis_lidar(self, **kwargs)

    def unload_data(self):
        self.points = None

    def has_bbs(self):
        labelPath = osp.join(self.seq_root, self.labelFolder, self.frame + ".txt")
        return osp.exists(labelPath)

    def dim(self):
        return self._dim


class Aeva(Lidar):
    def __init__(self, path):
        Lidar.__init__(self, path)
        self._dim = 10
    
    def load_data(self):
        self.points = load_lidar(self.path, dim=self._dim)
        return self.points


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
        self.chirp_type = None

    def load_data(self):
        # Loads polar radar data, timestamps, azimuths, and resolution value
        # Additionally, loads a pre-computed cartesian radar image and binary mask if they exist.
        self.timestamps, self.azimuths, self.chirp_type, self.polar, self.resolution = load_radar(
            self.path
        )
        cart_path = osp.join(self.sensor_root, "cart", self.frame + ".png")
        if osp.exists(cart_path):
            self.cartesian = cv2.imread(cart_path, cv2.IMREAD_GRAYSCALE)
        mask_path = osp.join(self.sensor_root, "mask", self.frame + ".png")
        if osp.exists(mask_path):
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return self.timestamps, self.azimuths, self.polar

    def unload_data(self):
        self.timestamps = None
        self.azimuths = None
        self.polar = None
        self.cartesian = None
        self.mask = None

    def polar_to_cart(
        self, cart_resolution, cart_pixel_width, polar=None, in_place=True
    ):
        """Converts a polar scan from polar to Cartesian format
        Args:
            cart_resolution (float): resolution of the output Cartesian image in (m / pixel)
            cart_pixel_width (int): width of the output Cartesian image in pixels
            polar (np.ndarray): if supplied, this function will use this input and not self.polar.
            in_place (bool): if True, self.cartesian is updated.
        """
        if polar is None:
            polar = self.polar
        cartesian = radar_polar_to_cartesian(
            self.azimuths, polar, self.resolution, cart_resolution, cart_pixel_width
        )
        if in_place:
            self.cartesian = cartesian
        return cartesian
    
    def undistort_radar_to_cart(
        self, query_poses, cart_resolution, cart_pixel_width, azimuth_upsample=4, polar=None, in_place=False 
    ):
        """Undistort motion of a polar radar scan into a Cartesian image using per-azimuth poses.
        Uses radar frame pose as the reference frame.

        Args:
            query_poses (List[np.ndarray]): list of 4x4 poses for each azimuth (T_v_i vehicle and inertial frames)
            cart_resolution (float): resolution of the output Cartesian image in (m / pixel)
            cart_pixel_width (int): width of the output Cartesian image in pixels
            azimuth_upsample (int): upsampling factor for number of azimuths via interpolation (1 = no interpolation)
            polar (np.ndarray): if supplied, this function will use this input and not self.polar.
            in_place (bool): if True, self.cartesian is updated.
        """
        if polar is None:
            polar = self.polar
        A, R = polar.shape
        T_ref_enu = np.linalg.inv(self.pose)

        # Cartesian image setup
        cart = np.zeros((cart_pixel_width, cart_pixel_width), dtype=np.float32)
        weight = np.zeros_like(cart)

        center = (cart_pixel_width - 1) / 2.0
        ranges = (np.arange(R, dtype=np.float32) + 0.5) * self.resolution # ranges of each bin

        def splat_ray(x_ref, y_ref, vals):
            col =  y_ref / cart_resolution + center
            row = center - x_ref / cart_resolution
            
            # Only keep values within cartesian image
            valid = (
                (col >= 0) & (col < cart_pixel_width - 1) &
                (row >= 0) & (row < cart_pixel_width - 1)
            )

            if not np.any(valid):
                return

            colv = col[valid]
            rowv = row[valid]
            valv = vals[valid]

            # Bilinear splat
            c0 = np.floor(colv).astype(np.int32)
            r0 = np.floor(rowv).astype(np.int32)
            dc = colv - c0
            dr = rowv - r0

            # Bilinear interpolation weights
            w00 = (1 - dc) * (1 - dr) # top-left
            w01 = dc * (1 - dr) # top-right
            w10 = (1 - dc) * dr # bottom-left
            w11 = dc * dr # bottom-right

            np.add.at(cart, (r0, c0), w00 * valv)
            np.add.at(weight,(r0, c0), w00)

            np.add.at(cart, (r0,c0 + 1), w01 * valv)
            np.add.at(weight, (r0, c0 + 1), w01)

            np.add.at(cart, (r0 + 1, c0), w10 * valv)
            np.add.at(weight, (r0 + 1, c0), w10)

            np.add.at(cart, (r0 + 1, c0 + 1), w11 * valv)
            np.add.at(weight, (r0 + 1, c0 + 1), w11)

        # Precompute all transformed rays once
        rays_x_ref = []
        rays_y_ref = []

        for i in range(A):
            theta = self.azimuths[i]
            T_ref_i = T_ref_enu @ query_poses[i]

            c = np.cos(theta)
            s = np.sin(theta)

            # Get cartesian points
            x_i = ranges * c
            y_i = ranges * s
            z_i = np.zeros_like(ranges)
            pts_i = np.stack([x_i, y_i, z_i], axis=0)

            # Transform into reference frame
            R_ref_i = T_ref_i[:3, :3]
            t_ref_i = T_ref_i[:3, 3:4]
            pts_ref = R_ref_i @ pts_i + t_ref_i

            rays_x_ref.append(pts_ref[0])
            rays_y_ref.append(pts_ref[1])

        # Interpolate between adjacent azimuths
        for i in range (A - 1):
            x0 = rays_x_ref[i]
            y0 = rays_y_ref[i]
            v0 = polar[i].astype(np.float32)

            x1 = rays_x_ref[i + 1]
            y1 = rays_y_ref[i + 1]
            v1 = polar[i + 1].astype(np.float32)

            # Upsample number of azimuths for interpolation
            for k in range(azimuth_upsample):
                alpha = k / azimuth_upsample # alpha is a linear interpolation factor
                x_ref = (1.0 - alpha) * x0 + alpha * x1
                y_ref = (1.0 - alpha) * y0 + alpha * y1
                vals  = (1.0 - alpha) * v0 + alpha * v1

                splat_ray(x_ref, y_ref, vals)

        # Need to include last original ray explicitly
        splat_ray(rays_x_ref[-1], rays_y_ref[-1], polar[-1].astype(np.float32))

        mask = weight > 1e-6
        cart[mask] /= weight[mask]

        return cart


    def visualize(self, **kwargs):
        return vis_radar(self, **kwargs)
