import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('tkagg')
import multiprocessing
from multiprocessing import Pool
from pyboreas.utils.utils import se3ToSE3, is_sorted
import open3d as o3d
import time
from pyboreas.data.himmelsbach import Himmelsbach

class PointCloud:
    """
    Class for working with (lidar) pointclouds.
    """
    def __init__(self, points):
        # points (np.ndarray): (N, 6) [x, y, z, intensity, laser_number, time]
        self.points = points

    def transform(self, T, in_place=True):
        """Transforms points given a transform, T. x_out = np.matmul(T, x)

        Args:
            T (np.ndarray): 4x4 transformation matrix
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): The transformed points
        """
        assert (T.shape[0] == 4 and T.shape[1] == 4)
        if in_place:
            points = self.points
        else:
            points = np.copy(self.points)
        p = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points[:, :3] = np.matmul(p, T.transpose())[:, :3]
        return points

    def remove_motion(self, body_rate, tref=None, in_place=True, single=False):
        """Removes motion distortion from a pointcloud

        Args:
            body_rate (np.ndarray): (6, 1) [vx, vy, vz, wx, wy, wz] in sensor frame
            tref (float): reference time to transform the points towards
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): points with motion distortion removed
        """
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

    def passthrough(self, bounds=[], in_place=True):
        """Removes points outside the specified bounds

        Args:
            bounds (list): [xmin, xmax, ymin, ymax, zmin, zmax]
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): the remaining points after the filter is applied
        """
        if len(bounds) < 6:
            print('Warning: len(bounds) = {} < 6 is incorrect!'.format(len(bounds)))
            return self.points
        p = self.points[np.where((self.points[:, 0] >= bounds[0]) &
                                 (self.points[:, 0] <= bounds[1]) &
                                 (self.points[:, 1] >= bounds[2]) &
                                 (self.points[:, 1] <= bounds[3]) &
                                 (self.points[:, 2] >= bounds[4]) &
                                 (self.points[:, 2] <= bounds[5]))]
        if in_place:
            self.points = p
        return p

    # Assumes pointcloud has already been transformed into the camera frame
    # color options: depth, intensity
    # returns pixel locations for lidar point projections onto an image plane
    def project_onto_image(self, P, width=2448, height=2048, color='depth', checkdims=True):
        """Projects 3D points onto a 2D image plane

        Args:
            P (np.ndarray): [fx 0 cx 0; 0 fy cy 0; 0 0 1 0; 0 0 0 1] cam projection
            width (int): width of image
            height (int): height of image
            color (str): 'depth' or 'intensity' to pick colors output
        Return:
            uv (np.ndarray): (N, 2) projected u-v pixel locations in an image
            colors (np.ndarray): (N,) a color value for each pixel location in uv.
            mask (np.ndarray): mask to select only points that project onto image.
        """
        colors = []
        x = np.hstack((self.points[:, :3], np.ones((self.points.shape[0], 1))))
        x /= x[:, 2:3]
        x[:, 3] = 1
        x = np.matmul(x, P.transpose())
        if checkdims:
            mask = np.where((x[:, 0] >= 0) &
                            (x[:, 0] <= width - 1) &
                            (x[:, 1] >= 0) &
                            (x[:, 1] <= height - 1))
        else:
            mask = np.ones(x.shape[0], dtype=np.bool)
        x = x[mask]
        if color == 'depth':
            colors = self.points[mask][:, 2]
        elif color == 'intensity':
            colors = self.points[mask][:, 3]
        else:
            print('Warning: {} is not a valid color'.format(color))
            colors = self.points[mask][:, 2]
        return x[:, :2], colors, mask

    def random_downsample(self, downsample_rate, in_place=True):
        rand_idx = np.random.choice(self.points.shape[0],
                                    size=int(self.points.shape[0] * downsample_rate),
                                    replace=False)
        p = self.points[rand_idx, :]
        if in_place:
            self.points = p
        return p

    def remove_ground_ransac(self, inlier_thresh=0.3, term_thresh=0.95, pts_range_xy=20, pts_range_z=0.5, max_iters=500,
                             in_place=True, show=False, show_subsample=False, print_debug=False):
        """Removes ground plane points via a RANSAC plane fit on a subsampled portion of the lidar pointcloud around the vehicle.

        Args:
            inlier_thresh (float): threshold to consider a point an inlier in the plane (m)
            term_thresh (float): inlier probability threshold to terminate the RANSNAC plane fit
            pts_range_xy (float): distance around lidar (in x and y axis) to clip pointcloud before doing plane fit (m)
            pts_range_z (float): distance around ground (in z axis) to clip pointcloud before doing plane fit (m)
            max_iters (int): max iterations for RANSAC algorithm
            in_place (bool): if True, self.points is updated
            show (bool): if True, generates a plot of the plane fit result
            show_subsample: (bool) if True, generates a plot of the subsampled lidar cloud
            print_debug (bool): if True, prints simple debug information

        Returns:
            points (nd.array): [n x 6] array of points with the ground plane removed
        """
        lidar_height_z = 2.13  # From calibration
        subsample = self.passthrough([-pts_range_xy, pts_range_xy,  # Only use a subsample of points around the vehicle for ransac plane fit
                                      -pts_range_xy, pts_range_xy,
                                      -pts_range_z - lidar_height_z, pts_range_z - lidar_height_z], in_place=False)[:, 0:3]
        max_inlier_pct = -1
        plane_norm = None

        for i in range(max_iters):
            rand_idx = np.random.choice(subsample.shape[0], 3)  # Choose 3 points for plane fit

            # Construct plane
            p_a = subsample[rand_idx[0]]
            p_b = subsample[rand_idx[1]]
            p_c = subsample[rand_idx[2]]
            v_a = p_a - p_b
            v_c = p_c - p_b
            norm = np.cross(v_a, v_c)
            u_norm = norm/np.linalg.norm(norm)

            # Get errors/inliers
            dists = np.abs((subsample - p_a) @ u_norm)
            inliers_count = (np.where(dists < inlier_thresh))[0].shape[0]
            inliers_pct = inliers_count / subsample.shape[0]
            if inliers_pct > max_inlier_pct:
                max_inlier_pct = inliers_pct
                plane_norm = u_norm
                dists_sub = dists
                if inliers_pct > term_thresh:  # Terminate if threshold met
                    break

        if print_debug:
            print(f"Iterations: {i+1} | Inliers% on Subsample: {max_inlier_pct:.4f}")

        dists = np.abs((self.points[:, 0:3] - [0, 0, -lidar_height_z]) @ plane_norm)
        outliers = self.points[np.where(dists > inlier_thresh)[0], :]

        if show_subsample:  # Plot the subsampled points used for plane fit
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(subsample)
            inlier_cloud = pcd.select_by_index(np.where(dists_sub < inlier_thresh)[0])
            inlier_cloud.paint_uniform_color([0, 0, 0])
            outlier_cloud = pcd.select_by_index(np.where(dists_sub < inlier_thresh)[0], invert=True)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Subsampled Pointcloud for GP Removal')

        if show:  # Plot inliers/outliers result of plane fit
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points[:, 0:3])
            inlier_cloud = pcd.select_by_index(np.where(dists < inlier_thresh)[0])
            inlier_cloud.paint_uniform_color([0, 0, 0])
            outlier_cloud = pcd.select_by_index(np.where(dists < inlier_thresh)[0], invert=True)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='GP Removal Results (Black = Ground)')

        if in_place:
            self.points = outliers
        return outliers


    def remove_ground_himmelsbach(self, show=False):

        lidar_height_z = 2.13  # From calibration

        himmel = Himmelsbach(self.points)
        himmel.set_alpha(2.0/180.0*np.pi)
        himmel.set_tolerance(0.25)
        himmel.set_thresholds(0.4, 0.2, 0.8, 5, 5)
        print("Himmelsbach initialized")
        t_himmel_start = time.time()
        ground_idx = himmel.compute_model_and_get_inliers()
        t_himmel_end = time.time()

        print("Himmelsbach finished in ", (t_himmel_end-t_himmel_start)*1000,'ms')

        if show:  # Plot inliers/outliers result of plane fit
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points[:, 0:3])
            ground_cloud = pcd.select_by_index(ground_idx)
            ground_cloud.paint_uniform_color([0, 0, 0])
            object_cloud = pcd.select_by_index(ground_idx, invert=True)
      
            o3d.visualization.draw_geometries([ground_cloud,object_cloud], window_name='GP Removal Results (Black = Ground)')


    def voxelize(self, voxel_size=0.1, show=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, 0:3])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=voxel_size)
        if show:
            o3d.visualization.draw_geometries([voxel_grid])
        return voxel_grid  # This is an o3d VoxelGrid object
