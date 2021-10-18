import numpy as np
import multiprocessing
from multiprocessing import Pool
from pyboreas.utils.utils import se3ToSE3

is_sorted = lambda x: (np.diff(x)>=0).all()

class PointCloud:
    def __init__(self, points):
        # x, y, z, (i, r, t)
        self.points = points

    def transform(self, T, in_place=True):
        assert (T.shape[0] == 4 and T.shape[1] == 4)
        if in_place:
            points = self.points
        else:
            points = np.copy(self.points)
        p = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points[:, :3] = np.matmul(p, T.transpose())[:, :3]
        return points

    def remove_motion(self, body_rate, tref=None, in_place=True):
        # body_rate: (6, 1) [vx, vy, vz, wx, wy, wz] in body frame
        assert(body_rate.shape[0] == 6 and body_rate.shape[1] == 1)
        tmin = np.min(self.points[:, 5])
        tmax = np.max(self.points[:, 5])
        if tref is None:
            tref = (tmin + tmax) / 2
        # Precompute finite number of transforms for speed
        bins = 21
        delta = (tmax - tmin) / float(bins - 1)
        T_undistorts = []
        tbins = []
        for i in range(bins):
            t = tmin + i * delta
            tbins.append(t)
            T_undistorts.append(se3ToSE3((t - tref) * body_rate))
        tbins[-1] += 1e-6

        if in_place:
            points = self.points
        else:
            points = np.copy(self.points)

        global _process_motion
        def _process_motion(i):
            index = int((points[i, 5] - tmin) / delta)
            pbar = np.vstack((points[i, :3].reshape(3, 1), 1))
            pbar = np.matmul(T_undistorts[index], pbar)
            points[i, :3] = pbar[:3, 0]

        # This function is ~10x faster if the pointcloud is sorted by timestamp (default)
        sections = []
        if is_sorted(points[:, 5]):
            for i in range(len(tbins) - 1):
                locs = np.where((points[:, 5] >= tbins[i]) & (points[:, 5] < tbins[i+1]))
                p = points[locs]
                p = np.hstack((p[:, :3], np.ones((p.shape[0], 1))))
                p = np.matmul(p, T_undistorts[i].transpose())
                p = np.hstack((p[:, :3], points[locs][:, 3:]))
                points[locs] = p
        else:
            p = Pool(multiprocessing.cpu_count())
            p.map(_process_motion, list(range(points.shape[0])))

        if in_place:
            self.points = points

        return points

    # bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    def passthrough(self, bounds=[], in_place=True):
        if len(bounds) < 6:
            print('Warning: len(bounds) = {} < 6 is incorrect!'.format(len(bounds)))
            return self.points
        p = self.points[np.where((self.points[:, 0] >= bounds[0]) & \
                                 (self.points[:, 0] <= bounds[1]) & \
                                 (self.points[:, 1] >= bounds[2]) & \
                                 (self.points[:, 1] <= bounds[3]) & \
                                 (self.points[:, 2] >= bounds[4]) & \
                                 (self.points[:, 2] <= bounds[5]))]
        if in_place:
            self.points = p
        return p

    # Assumes pointcloud has already been transformed into the camera frame
    # color options: depth, intensity
    # returns pixel locations for lidar point projections onto an image plane
    def project_onto_image(self, P, width=2448, height=2048, color='depth'):
        uv = []
        colors = []
        x = np.hstack((self.points[:, :3], np.ones((self.points.shape[0], 1))))
        x /= x[:, 2:3]
        x[:, 3] = 1
        x = np.matmul(x, P.transpose())
        mask = np.where((x[:, 0] >= 0) & \
                        (x[:, 0] <= width - 1) & \
                        (x[:, 1] >= 0) & \
                        (x[:, 1] <= height - 1))
        x = x[mask]
        if color == 'depth':
            colors = self.points[mask][:, 2]
        elif color == 'intensity':
            colors = self.points[mask][:, 3]
        else:
            print('Warning: {} is not a valid color'.format(color))
            colors = np.ones(x.shape[0])
        return x[:, :2], colors

# TODO: remove_ground(self, bool: in_place)
# TODO: voxelize(self)