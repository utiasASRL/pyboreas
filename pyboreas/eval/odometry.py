import argparse
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


from pyboreas.utils.odometry import (
    compute_kitti_metrics,
    get_sequence_poses,
    get_sequence_poses_gt,
    get_sequences,
    get_sequence_velocities,
    get_sequence_velocities_gt,
    compute_vel_metrics
)


def eval_odom(pred="test/demo/pred/3d", gt="test/demo/gt", radar=False, velocity=False):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    pred_pose = osp.join(pred, "odometry_result")
    seq = get_sequences(pred_pose, ".txt")

    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred_pose, seq)

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred_pose, dim, crop
    )

    if velocity:
        # parse sequences
        pred_vel_path = osp.join(pred, "odometry_vel_result")
        seq = get_sequences(pred_vel_path, ".txt")

        vel_pred, times_pred, seq_vel_lens_pred = get_sequence_velocities(pred_vel_path, seq, dim)

        # get corresponding groundtruth poses
        vel_gt, _, seq_lens_gt, crop = get_sequence_velocities_gt(gt, seq, dim)

        # compute errors
        v_RMSE, v_mean, v_RMSE_out, v_mean_out = compute_vel_metrics(vel_gt, vel_pred, times_pred, seq, pred_vel_path, dim, crop)

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error: ", t_err, " %, ", r_err, " deg/m")

    if velocity:
        if dim == 2:
            print("Velocity RMSE: ", v_RMSE, " [m/s, m/s, deg/s]")
            print("Velocity mean: ", v_mean, " [m/s, m/s, deg/s]")
            print("Velocity RMSE w/o outliers: ", v_RMSE_out, " [m/s, m/s, deg/s]")
            print("Velocity mean w/o outliers: ", v_mean_out, " [m/s, m/s, deg/s]")
        else:
            print("Velocity RMSE: ", v_RMSE, " [m/s, m/s, m/s, deg/s, deg/s, deg/s]")
            print("Velocity mean: ", v_mean, " [m/s, m/s, m/s, deg/s, deg/s, deg/s]")
            print("Velocity RMSE w/o outliers: ", v_RMSE_out, " [m/s, m/s, m/s, deg/s, deg/s, deg/s]")
            print("Velocity mean w/o outliers: ", v_mean_out, " [m/s, m/s, m/s, deg/s, deg/s, deg/s]")


    return t_err, r_err


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", default="test/demo/pred/3d", type=str, help="path to prediction files"
    )
    parser.add_argument(
        "--gt", default="test/demo/gt", type=str, help="path to groundtruth files"
    )
    parser.add_argument(
        "--radar",
        dest="radar",
        action="store_true",
        help="evaluate radar odometry in SE(2)",
    )

    parser.add_argument(
        "--velocity",
        dest="velocity",
        action="store_true",
        help="evaluate velocity (default: False)",
    )

    parser.set_defaults(radar=False)
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.radar, args.velocity)
