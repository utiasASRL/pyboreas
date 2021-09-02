import numpy as np
from utils.utils import carrot

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

    def remove_motion(self, points, body_rate, tref=None, in_place=True):
        # body_rate: (6, 1) [vx, vy, vz, wx, wy, wz] in body frame
        # Note: modifies points contained in this class
        assert (body_rate.shape[0] == 6 and body_rate.shape[1] == 1)
        tmin = np.min(points[:, 5])
        tmax = np.max(points[:, 5])
        if tref is None:
            tref = (tmin + tmax) / 2
        # Precompute transforms for compute speed
        bins = 101
        delta = (tmax - tmin) / (bins - 1)
        T_undistorts = []
        for i in range(bins):
            t = tmin + i * delta
            T_undistorts.append((t - tref) * carrot(body_rate))
        if not in_place:
            ptemp = np.copy(points)
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