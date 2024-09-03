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


def eval_odom(pred="test/demo/pred/3d", gt="test/demo/gt", radar=False):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    # Addition to filter parameter specific sequence to regular sequence ground truth
    seq_gt = []
    for s in seq:
        curr_seq = s.split('_')[0]
        if ".txt" != curr_seq[-4:]: 
            curr_seq=curr_seq+".txt"
        seq_gt.append(curr_seq)

    # seq_gt = [s.split('_')[0]+".txt" for s in seq]

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq_gt, dim)

    # compute errors
    t_err, r_err, _, t_re_rmse, t_re_rmse_99f9 = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop
    )

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error: ", t_err, " %, ", r_err, " deg/m")

    # return t_err, r_err
    return t_err, r_err, t_re_rmse, t_re_rmse_99f9


def eval_odom_vel(pred="test/demo/pred/3d", gt="test/demo/gt", radar=False):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    seq = get_sequences(pred, ".txt")
    vel_pred, times_pred, seq_vel_lens_pred = get_sequence_velocities(pred, seq, dim)

    # Addition to filter parameter specific sequence to regular sequence ground truth
    seq_gt = []
    for s in seq:
        curr_seq = s.split('_')[0]
        if ".txt" != curr_seq[-4:]: 
            curr_seq=curr_seq+".txt"
        seq_gt.append(curr_seq)
    
    # seq_gt = [s.split('_')[0]+".txt" for s in seq]

    # get corresponding groundtruth poses
    # vel_gt, _, seq_lens_gt, crop = get_sequence_velocities_gt(gt, seq, dim)
    vel_gt, _, seq_lens_gt, crop = get_sequence_velocities_gt(gt, seq_gt, dim)

    # compute errors
    v_RMSE, v_mean, v_RMSE_out, v_mean_out = compute_vel_metrics(vel_gt, vel_pred, times_pred, seq, pred, dim, crop)

    # print out results
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

    return v_RMSE, v_mean


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
        "--velocity", default=None, type=str, help="path to prediction files"
    )

    parser.set_defaults(radar=False)
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.radar)
    if args.velocity is not None:
        eval_odom_vel(args.velocity, args.gt, args.radar)
