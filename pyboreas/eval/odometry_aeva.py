import argparse
import os
from itertools import accumulate
from time import time

import numpy as np

from pyboreas.utils.odometry import (
    calc_sequence_errors,
    get_sequence_poses,
    get_sequences,
    get_stats,
    plot_stats,
    read_traj_file_gt,
)


def get_sequence_poses_gt(path, seq):
    """Retrieves a list of the poses corresponding to the given sequences in the given file path with the Boreas dataset
    directory structure.
    Args:
        path (string): directory path to root directory of Boreas dataset
        seq (List[string]): list of sequence file names
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        all_poses (List[np.ndarray]): list of 4x4 poses from all sequence files
        all_times (List[int]): list of times in microseconds from all sequence files
        seq_lens (List[int]): list of sequence lengths
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    """

    # loop for each sequence
    all_poses = []
    all_times = []
    seq_lens = []
    for filename in seq:
        # determine path to gt file
        dir = filename[:-4]  # assumes last four characters are '.txt'

        filepath = os.path.join(
            path, dir, "applanix/aeva_poses.csv"
        )  # use 'aeva_poses.csv' for groundtruth
        T_calib = np.loadtxt(os.path.join(path, dir, "calib/T_applanix_aeva.txt"))
        poses, times = read_traj_file_gt(filepath, T_calib, 3)

        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)

    return all_poses, all_times, seq_lens


def compute_kitti_metrics(
    T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, plot_dir, dim, crop
):
    """Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        KITTI rotation and translation metrics are computed for each sequence individually and then
        averaged across the sequences. If 'interp' specifies a directory, we instead interpolate
        for poses at the groundtruth times and write them out as txt files.
    Args:
        T_gt (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        T_pred (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        seq_lens_gt (List[int]): List of sequence lengths corresponding to T_gt
        seq_lens_pred (List[int]): List of sequence lengths corresponding to T_pred
        seq (List[string]): List of sequence file names
        plot_dir (string): path to output directory for plots. Set to '' (empty string) to prevent plotting
        dim (int): dimension for evaluation. Set to '3' for SE(3) or '2' for SE(2)
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    Returns:
        t_err: Average KITTI Translation ERROR (%)
        r_err: Average KITTI Rotation Error (deg / m)
    """
    # set step size
    if dim == 3:
        step_size = 10  # every 10 frames should be 1 second
    elif dim == 2:
        step_size = 4  # every 4 frames should be 1 second
    else:
        raise ValueError(
            "Invalid dim value in compute_kitti_metrics. Use either 2 or 3."
        )

    # get start and end indices of each sequence
    indices_gt = [0]
    indices_gt.extend(list(accumulate(seq_lens_gt)))
    indices_pred = [0]
    indices_pred.extend(list(accumulate(seq_lens_pred)))

    # loop for each sequence
    err_list = []
    for i in range(len(seq_lens_pred)):
        ts = time()  # start time

        # get poses and times of current sequence
        T_gt_seq = T_gt[indices_gt[i] : indices_gt[i + 1]]
        T_pred_seq = T_pred[indices_pred[i] : indices_pred[i + 1]]
        # times_gt_seq = times_gt[indices_gt[i]:indices_gt[i+1]]
        # times_pred_seq = times_pred[indices_pred[i]:indices_pred[i+1]]

        print("processing sequence", seq[i], "...")

        # 2d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size, 2)
        t_err_2d, r_err_2d, _, _ = get_stats(err, path_lengths)

        # 3d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size)
        t_err, r_err, t_err_len, r_err_len = get_stats(err, path_lengths)

        print(seq[i], "took", str(time() - ts), " seconds")
        # print('Error: ', t_err, ' %, ', r_err, ' deg/m \n')
        print(f"& {t_err_2d:.2f} & {r_err_2d:.4f} & {t_err:.2f} & {r_err:.4f} \\\\")

        err_list.append([t_err, r_err])

        if plot_dir:
            plot_stats(
                seq[i],
                plot_dir,
                T_pred_seq,
                T_gt_seq,
                path_lengths,
                t_err_len,
                r_err_len,
            )

    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]

    return t_err, r_err, err_list


def eval_odom(pred, gt):
    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt = get_sequence_poses_gt(gt, seq)

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, 3, None
    )

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error: ", t_err, " %, ", r_err, " deg/m")
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
    args = parser.parse_args()

    eval_odom(args.pred, args.gt)
