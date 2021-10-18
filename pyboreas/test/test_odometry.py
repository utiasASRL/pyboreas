"""Tests for odometry benchmark"""
import unittest
import numpy as np

from pylgmath import Transformation, se3op
from utils.utils import get_inverse_tf
from utils.odometry import interpolate_poses, write_traj_file, read_traj_file


class OdometryTestCase(unittest.TestCase):
    """This class contains tests for the odometry benchmark script."""
    def test_interpolate(self):
        """Checks the pysteam interpolation against a known constant-velocity groundtruth."""
        # setup groundtruth
        delt = 0.05
        num_poses = 50

        # constant velocity
        v_x = -1.0
        omega_z = 0.01
        velocity_prior = np.array([[v_x, 0., 0., 0., 0., omega_z]]).T
        init_pose_vec = np.array([[0., 0., 0., 0., 0., 0.]]).T
        init_pose = Transformation(xi_ab=init_pose_vec)

        # create trajectory
        poses = [se3op.vec2tran(i*delt*velocity_prior)
                 @ init_pose.matrix() for i in range(num_poses)]
        times = [int(i*delt*1e9) for i in range(num_poses)]     # nanoseconds

        query_poses = interpolate_poses(poses[::2], times[::2], times[1:-1:2])
        gt_poses = poses[1:-1:2]
        for query_pose, gt_pose in zip(query_poses, gt_poses):
            delta = gt_pose@get_inverse_tf(query_pose)
            self.assertTrue(np.linalg.norm(se3op.tran2vec(delta)) < 1e-6)

    def test_read_write(self):
        """Checks read and write functions for trajectory."""
        # setup groundtruth
        delt = 0.05
        num_poses = 50

        # constant velocity
        v_x = -1.0
        omega_z = 0.01
        velocity_prior = np.array([[v_x, 0., 0., 0., 0., omega_z]]).T
        init_pose_vec = np.array([[0., 0., 0., 0., 0., 0.]]).T
        init_pose = Transformation(xi_ab=init_pose_vec)

        # create trajectory
        write_poses = [se3op.vec2tran(i*delt*velocity_prior)
                       @ init_pose.matrix() for i in range(num_poses)]
        write_times = [int(i*delt*1e9) for i in range(num_poses)]     # nanoseconds

        # write out trajectory to file, then read it back
        write_traj_file('test/test_traj.txt', write_poses, write_times)
        read_poses, read_times = read_traj_file('test/test_traj.txt')

        # compare write and read
        for wpose, wtime, rpose, rtime in zip(write_poses, write_times, read_poses, read_times):
            delta = wpose@get_inverse_tf(rpose)
            self.assertTrue(np.linalg.norm(se3op.tran2vec(delta)) < 1e-6)
            self.assertTrue(wtime == rtime)


if __name__ == '__main__':
    # run 'python -m unittest' from 'boreas-devkit/python' directory
    unittest.main()
