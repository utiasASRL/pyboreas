import numpy as np
from pyboreas.utils.utils import se3ToSE3

class PointCloud:
    def __init__(self, points):
        # x, y, z, (i, r, t)
        self.points = points

    def transform(self, T):
        assert (T.shape[0] == 4 and T.shape[1] == 4)
        for i in range(self.points.shape[0]):
            pbar = np.vstack((self.points[i, :3], np.array([1])))
            pbar = np.matmul(T, pbar)
            self.points[i, :3] = pbar[:3, 0]

    def remove_motion(self, body_rate, tref=None, in_place=True):
        # body_rate: (6, 1) [vx, vy, vz, wx, wy, wz] in body frame
        assert(body_rate.shape[0] == 6 and body_rate.shape[1] == 1)
        tmin = np.min(self.points[:, 5])
        tmax = np.max(self.points[:, 5])
        if tref is None:
            tref = (tmin + tmax) / 2
        # Precompute finite number of transforms for speed
        bins = 101
        delta = (tmax - tmin) / float(bins - 1)
        T_undistorts = []
        for i in range(bins):
            t = tmin + i * delta
            T_undistorts.append(se3ToSE3((t - tref) * body_rate))
        if not in_place:
            ptemp = np.copy(self.points)
        for i in range(self.points.shape[0]):
            pbar = np.vstack((self.points[i, :3], np.array([1])))
            index = int((self.points[i, 5] - tmin) / delta)
            pbar = np.matmul(T_undistorts[index], pbar)
            if in_place:
                self.points[i, :3] = pbar[:3, 0]
            else:
                ptemp[i, :3] = pbar[:3, 0]
        if not in_place:
            return ptemp

# TODO: remove_ground(self, bool: in_place)