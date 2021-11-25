"""Tests for odometry benchmark"""
import os
import unittest
import numpy as np
import math

from pylgmath import Transformation, se3op
from pyboreas.utils.utils import get_inverse_tf
from pyboreas.utils.odometry import interpolate_poses, write_traj_file, read_traj_file, \
    get_sequences, get_sequence_poses, get_sequence_poses_gt, compute_kitti_metrics, compute_interpolation


class OdometryTestCase(unittest.TestCase):
    """This class contains tests for the odometry benchmark script."""
    def test_interpolate(self):
        """Checks the pysteam interpolation against a known constant-velocity groundtruth."""
        # setup groundtruth
        delt = 0.05     # seconds
        num_poses = 50  # total frames

        # constant velocity
        v_x = -1.0
        omega_z = 0.01
        velocity_prior = np.array([[v_x, 0., 0., 0., 0., omega_z]]).T
        init_pose_vec = np.array([[0., 0., 0., 0., 0., 0.]]).T
        init_pose = Transformation(xi_ab=init_pose_vec)

        # create trajectory
        poses = [se3op.vec2tran(i*delt*velocity_prior)
                 @ init_pose.matrix() for i in range(num_poses)]
        times = [int(i*delt*1e6) for i in range(num_poses)]     # microseconds

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
        write_times = [int(i*delt*1e6) for i in range(num_poses)]     # microseconds

        # write out trajectory to file, then read it back
        write_traj_file('pyboreas/test/test_traj.txt', write_poses, write_times)
        read_poses, read_times = read_traj_file('pyboreas/test/test_traj.txt')

        # compare write and read
        for wpose, wtime, rpose, rtime in zip(write_poses, write_times, read_poses, read_times):
            delta = wpose@get_inverse_tf(rpose)
            self.assertTrue(np.linalg.norm(se3op.tran2vec(delta)) < 1e-6)
            self.assertTrue(wtime == rtime)

        # delete file
        os.remove('pyboreas/test/test_traj.txt')

    def test_module(self):
        pred = 'pyboreas/test/demo/pred/3d'
        gt = 'pyboreas/test/demo/gt'
        dim = 3
        interp = 'pyboreas/test/demo/pred/3d/interptest'
        solver = False
        processes = 2

        # make interp directory if it doesn't exist
        if not os.path.exists(interp):
            os.mkdir(interp)

        # parse sequences
        seq = get_sequences(pred, '.txt')
        T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)
        T_gt, times_gt, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

        # interpolate
        compute_interpolation(T_pred, times_gt, times_pred, seq_lens_gt, seq_lens_pred, seq, interp, solver, processes)

        # read in interpolated sequences
        T_pred, times_pred, seq_lens_pred = get_sequence_poses(interp, seq)

        # compute errors
        _, _, err_list = compute_kitti_metrics(T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, '', dim, crop)

        # first sequence should be close to kitti C++ results: 0.011094 0.000077
        self.assertTrue(math.fabs(err_list[0][0] - 0.011094*100) < 1e-4)
        self.assertTrue(math.fabs(err_list[0][1] - 0.000077*180/math.pi) < 1e-5)

        # second sequence should be close to 0
        self.assertTrue(err_list[1][0] < 1e-1)
        self.assertTrue(err_list[1][1] < 1e-2)

        # delete file
        for i in range(2):
            file = os.path.join(interp, seq[i])
            os.remove(file)
        os.rmdir(interp)


if __name__ == '__main__':
    # run 'python -m unittest' from root directory (one above pyboreas directory)
    unittest.main()
