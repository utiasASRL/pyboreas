import argparse
import os
from itertools import accumulate
from time import time

import numpy as np
import pandas as pd

from pyboreas.utils.odometry import (
    calc_sequence_errors,
    convert_line_to_pose,
    get_sequence_poses,
    get_sequences,
    get_stats,
    plot_stats,
)

from pyboreas.utils.utils import (
    enforce_orthog,
    get_inverse_tf,
    tranAd,
    se3ToSE3,
    SE3Tose3,
)

import pyboreas.utils.se3_utils_numpy as se3


def get_sequence_poses_gt(path, seq, data_type):
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

        if (data_type == "aeva_hq"):
            filepath = os.path.join(path, dir, "processed_sbet.csv") # T_world_applanix
            T_r_app = np.array([[ 9.99960818e-01,-1.40913767e-03, 8.73943838e-03, 0.00000000e+00],
                                [ 1.40913767e-03, 9.99999007e-01, 6.15750237e-06, 0.00000000e+00],
                                [-8.73943838e-03, 6.15781076e-06, 9.99961810e-01, 0.00000000e+00],
                                [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]).astype(np.float64)
            T_ref = np.array([[1, 0, 0, 0], 
                              [0,-1, 0, 0], 
                              [0, 0,-1, 0], 
                              [0, 0, 0, 1]])
            T_calib = np.eye(4) # will convert to vehicle frame after interpolation
        elif (data_type == "aeva_boreas"):
            filepath = os.path.join(path, dir, "applanix/aeva_poses.csv")  # use 'aeva_poses.csv' for groundtruth, T_world_sensor
            T_as = np.loadtxt(os.path.join(path, dir, "calib/T_applanix_aeva.txt"))
            T_sr = np.array([[ 0.9999366830849237  , 0.008341717781538466 , 0.0075534496251198685,-1.0119098938516395],
                             [-0.008341717774127972, 0.9999652112886684   ,-3.150635091210066e-05,-0.3965882433517194],
                             [-0.007553449599178521,-3.150438868196706e-05, 0.9999714717963843   ,-1.697000000000001 ],
                             [ 0.00000000e+00      , 0.00000000e+00       , 0.00000000e+00       , 1.00000000e+00    ]]).astype(np.float64)
            T_calib = T_sr
        
        with open(filepath, "r") as f:
            lines = f.readlines()
        poses = []
        times = []

        T_ab = enforce_orthog(T_calib)
        for line in lines[1:]:
            pose, time = convert_line_to_pose(line, 3)
            poses += [
                enforce_orthog(get_inverse_tf(pose @ T_calib))
            ]  # convert T_iv to T_vi and apply calibration
            times += [int(time)]  # microseconds

        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)

    return all_poses, all_times, seq_lens

def pdcsv_to_se3_pose(pdcsv):
    Tis = np.zeros((pdcsv.shape[0], 4, 4), dtype=np.float64)
    Tis[:, -1, -1] = 1
    Tis[:, :3, :3] = se3.ypr2rot(pdcsv.loc[:, 'roll'], pdcsv.loc[:, 'pitch'], pdcsv.loc[:, 'heading'])
    Tis[:, 0, -1] = pdcsv.loc[:, 'easting']
    Tis[:, 1, -1] = pdcsv.loc[:, 'northing']
    Tis[:, 2, -1] = pdcsv.loc[:, 'altitude']
    return Tis

def pdcsv_to_body_vel(pdcsv, Tis):
    w = np.zeros((pdcsv.shape[0], 3), dtype=np.float64)    # angular (already in sensor frame)
    v = np.zeros((pdcsv.shape[0], 3), dtype=np.float64)    # translational    (in inertial frame)
    w[:, 0] = pdcsv.loc[:, 'angvel_x']
    w[:, 1] = pdcsv.loc[:, 'angvel_y']
    w[:, 2] = pdcsv.loc[:, 'angvel_z']
    v[:, 0] = pdcsv.loc[:, 'vel_east']
    v[:, 1] = pdcsv.loc[:, 'vel_north']
    v[:, 2] = pdcsv.loc[:, 'vel_up']
    v = Tis[:, :3, :3].transpose(0, 2, 1) @ v[:, :, None] 
    return -np.concatenate((v[:, :, 0], w), axis=1)

def compute_kitti_metrics(T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, plot_dir, dim, crop):
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
        print(
            f"Terr(2D) {t_err_2d:.2f}%  Rerr(2D) {r_err_2d:.4f}deg/m  Terr(3D) {t_err:.2f}% Rerr(2D) {r_err:.4f}deg/m \\\\"
        )

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

# QOL: make gt data the same length as pred data
def adjust_length(T, seq_lens, target_len_T, target_len_seq_lens):
    if len(T) > target_len_T:
        T = T[:target_len_T]
        seq_lens = seq_lens[:target_len_seq_lens]
    elif len(T) < target_len_T:
        # pad with the last pose repeated
        last_pose = T[-1]
        pad_len = target_len_T - len(T)
        T += [last_pose] * pad_len
        seq_lens += [seq_lens[-1]] * pad_len
    return T, seq_lens

def utc_to_nanos_of_week(times_sec):
    gps_sec = times_sec - 315964800 + 18
    gps_ns = (gps_sec * 1e9).astype(int)
    ns_per_week = 60.0 * 60.0 * 24.0 * 7.0 * 1e9
    gps_week_number = (gps_ns/ns_per_week).astype(int)
    ns_to_start_of_week = gps_week_number * ns_per_week

    return np.mod(gps_ns, ns_to_start_of_week)

def wnoa_interp(time1, T1, w1, time2, T2, w2, timeq):
    tau = timeq - time1
    dt = time2 - time1
    ratio = tau / dt
    ratio2 = ratio * ratio
    ratio3 = ratio2 * ratio
    
    # Calculate 'psi' interpolation values
    psi11 = 3.0 * ratio2 - 2.0 * ratio3
    psi12 = tau * (ratio2 - ratio)
    psi21 = 6.0 * (ratio - ratio2) / dt
    psi22 = 3.0 * ratio2 - 2.0 * ratio

    # Calculate 'lambda' interpolation values
    lambda11 = 1.0 - psi11
    lambda12 = tau - dt * psi11 - psi12
    lambda21 = -psi21
    lambda22 = 1.0 - dt * psi21 - psi22

    xi_21 = se3.se3_log(T2 @ se3.se3_inv(T1)[0])
    J_21_inv = se3.se3_inv_left_jacobian(xi_21)
    xi_i1 = lambda12 * w1 + psi11 * xi_21 + psi12 * J_21_inv @ w2
    T_i1 = se3.se3_exp(xi_i1)[0]
    return T_i1 @ T1, np.zeros([6]) # temporary dummy velocity return 

def wnoa_interp_traj(src_times, T, w, qtimes):
    outT_list = []
    outw_list = []
    tail = 0
    head = 1
    for ts in qtimes:
        while not src_times[tail] <= ts <= src_times[head]:
            tail += 1
            head += 1
        assert(src_times[tail] <= ts)
        assert(src_times[head] >= ts)
        Tinterp, winterp = wnoa_interp(src_times[tail], T[tail], w[tail], src_times[head], T[head], w[head], ts)
        outT_list += [Tinterp]
        outw_list += [winterp]

    return np.array(outT_list), np.array(outw_list)

def get_aeva_hq_groundtruth(pdcsv, times_pred):
    seq_lens_gt = []
    T_r_app = np.array([[ 9.99960818e-01,-1.40913767e-03, 8.73943838e-03, 0.00000000e+00],
                        [ 1.40913767e-03, 9.99999007e-01, 6.15750237e-06, 0.00000000e+00],
                        [-8.73943838e-03, 6.15781076e-06, 9.99961810e-01, 0.00000000e+00],
                        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]).astype(np.float64)
    T_ref = np.array([[1, 0, 0, 0], 
                      [0,-1, 0, 0], 
                      [0, 0,-1, 0], 
                      [0, 0, 0, 1]])
    
    query_times = utc_to_nanos_of_week(np.array(times_pred) / 1e6) * 1e-9
    
    raw_pose = pdcsv_to_se3_pose(pdcsv)
    raw_varpi = pdcsv_to_body_vel(pdcsv, raw_pose)
    times_gt = pdcsv.loc[:, 'GPSTime'].to_numpy()
    
    # in body frame
    varpi_gt = (tranAd(T_ref) @ raw_varpi[:, :, None])[:, :, 0]
    raw_Tiv = raw_pose @ get_inverse_tf(T_ref)
        
    raw_Tiv = se3.se3_inv(raw_Tiv)
    
    # interpolate at pred times
    assert(query_times[0] > times_gt[0])
    assert(query_times[-1] < times_gt[-1])
    T_gt, _ = wnoa_interp_traj(times_gt, raw_Tiv, varpi_gt, query_times)
    T_gt = T_r_app @ T_gt
    seq_lens_gt.append(len(T_gt))
    return T_gt, times_gt, seq_lens_gt
    

def eval_odom(pred, gt, data_type):
    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)
    
    # get corresponding groundtruth poses
    if (data_type == "aeva_boreas"):
        T_gt, times_gt, seq_lens_gt = get_sequence_poses_gt(gt, seq, data_type)
    
    if (data_type == "aeva_hq"):
        for filename in seq:
            dir = filename[:-4]
            break
        pdcsv = pd.read_csv(os.path.join(gt, dir, "processed_sbet.csv"))
        T_gt, _, seq_lens_gt = get_aeva_hq_groundtruth(pdcsv, times_pred)
        
    print("sequence lengths (gt, pred): ", len(T_gt), ", ", len(T_pred))

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
    parser.add_argument(
        "--data_type", default="aeva_boreas", type=str, help="name of dataset"
    )
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.data_type)
