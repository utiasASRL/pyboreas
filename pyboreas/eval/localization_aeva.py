import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np

from pyboreas.data.splits import loc_reference
from pyboreas.utils.odometry import plot_loc_stats, read_traj_file2, read_traj_file_gt2
from pyboreas.utils.utils import (
    SE3Tose3,
    get_closest_index,
    get_inverse_tf,
    rotToRollPitchYaw,
)


def get_Tas(gtpath, seq, sensor="aeva"):
    T_applanix_aeva = np.loadtxt(osp.join(gtpath, seq, 'calib', 'T_applanix_aeva.txt'))
    if sensor == "camera":
        T_camera_lidar = np.loadtxt(osp.join(gtpath, seq, "calib", "T_camera_lidar.txt"))
        return np.matmul(T_applanix_lidar, get_inverse_tf(T_camera_lidar))
    elif sensor == "radar":
        T_radar_lidar = np.loadtxt(osp.join(gtpath, seq, "calib", "T_radar_lidar.txt"))
        return np.matmul(T_applanix_lidar, get_inverse_tf(T_radar_lidar))
    elif sensor == "lidar":
        T_applanix_lidar = np.loadtxt(osp.join(gtpath, seq, "calib", "T_applanix_lidar.txt"))
        return T_applanix_lidar
    return T_applanix_aeva


def check_time_match(pred_times, gt_times):
    assert len(pred_times) == len(
        gt_times
    ), f"pred time {len(pred_times)} is not equal to gt time {len(gt_times)}"
    p = np.array(pred_times)
    g = np.array(gt_times)
    assert np.sum(p - g) == 0


def check_ref_time_match(ref_times, gt_ref_times):
    indices = np.searchsorted(gt_ref_times, ref_times)
    p = np.array(ref_times)
    g = np.array(gt_ref_times)
    assert np.sum(g[indices] - p) == 0, f"{g[indices].shape} and {p.shape}"


def get_T_enu_s1(query_time, gt_times, gt_poses):
    closest = get_closest_index(query_time, gt_times)
    assert query_time == gt_times[closest], "query: {}".format(query_time)
    return gt_poses[closest]


def compute_errors(Te):
    r, p, y = rotToRollPitchYaw(Te[:3, :3])
    return [
        Te[0, 3],
        Te[1, 3],
        Te[2, 3],
        r * 180 / np.pi,
        p * 180 / np.pi,
        y * 180 / np.pi,
    ]


def root_mean_square(errs):
    return np.sqrt(np.mean(np.power(np.array(errs), 2), axis=0)).squeeze()

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

def eval_local(
    predpath,
    gtpath,
    gt_ref_seq,
    ref_sensor="aeva",
    test_sensor="aeva",
    dim=3,
    plot_dir=None,
):
    
    T_s_v = np.array([[0.9999366830849237, 0.008341717781538466, 0.0075534496251198685, -1.0119098938516395],
                        [-0.008341717774127972, 0.9999652112886684, -3.150635091210066e-05, -0.3965882433517194],
                        [-0.007553449599178521, -3.1504388681967066e-05, 0.9999714717963843, -1.697000000000001],
                        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]).astype(np.float64)
    
    pred_files = sorted(
        [
            f
            for f in os.listdir(predpath)
            if f.startswith("boreas-20") and f.endswith(".txt") and "err" not in f
        ]
    )
    gt_seqs = []
    for predfile in pred_files:
        if Path(predfile).stem.split(".")[0] not in os.listdir(gtpath):
            raise Exception(
                f"prediction file {predfile} doesn't match ground truth sequence list"
            )
        gt_seqs.append(Path(predfile).stem.split(".")[0])

    gt_ref_poses, gt_ref_times = read_traj_file_gt2(
        osp.join(gtpath, gt_ref_seq, "applanix", ref_sensor + "_poses.csv"), dim=dim
    )
    seq_rmse = []
    seq_consist = []
    seqs_have_cov = True
    for predfile, seq in zip(pred_files, gt_seqs):
        print("Processing {}...".format(seq))
        T_as = get_Tas(gtpath, seq, ref_sensor) # T_applanix_sensor
        T_sa = get_inverse_tf(T_as)             # T_sensor_applanix
        pred_poses, pred_times, ref_times, cov_matrices, has_cov = read_traj_file2(
            osp.join(predpath, predfile)
        )
        seqs_have_cov *= has_cov
        gt_poses, gt_times = read_traj_file_gt2(
            osp.join(gtpath, seq, "applanix", test_sensor + "_poses.csv"), dim=dim
        )
        
        # adjust the length of T_gt and seq_lens_gt
        gt_poses, gt_times = adjust_length(gt_poses, gt_times, len(pred_poses), len(pred_times))
        
        print(len(gt_poses), len(pred_poses))
        
        # check that pred_times is a 1-to-1 match with gt_times
        check_time_match(pred_times, gt_times)
        # check that each ref time matches to one gps_ref_time
        check_ref_time_match(ref_times, gt_ref_times)
        errs = []
        consist = []
        T_gt_seq = []
        T_pred_seq = []
        Xi = []
        Cov = []
        for j, pred_T_s1_s2 in enumerate(pred_poses):
            gt_T_enu_s2 = gt_poses[j]
            T_gt_seq.append(get_inverse_tf(gt_T_enu_s2))

            gt_T_enu_s1 = get_T_enu_s1(ref_times[j], gt_ref_times, gt_ref_poses)
            T_pred_seq.append(get_inverse_tf(gt_T_enu_s1 @ pred_T_s1_s2))

            gt_T_s1_s2 = get_inverse_tf(gt_T_enu_s1) @ gt_T_enu_s2
            T = pred_T_s1_s2 @ get_inverse_tf(gt_T_s1_s2)
            Te = T_as @ T @ T_sa # error is reported with x lateral, y longitudinal
            errs.append(compute_errors(Te))
            # If the user submitted a covariance matrix, calculate consistency
            if has_cov:
                if abs(np.sum(cov_matrices[j] - np.identity(6))) < 1e-3:
                    consist.append(1)
                xi = SE3Tose3(T)
                Xi.append(xi.squeeze())
                c = xi.T @ cov_matrices[j] @ xi
                consist.append(
                    c[0, 0]
                )  # assumes user has uploaded inverse covariance matrices
                Cov.append(1 / cov_matrices[j].diagonal())
        Xi = np.array(Xi)
        Cov = np.array(Cov)
        if plot_dir is not None:
            plot_err_file = osp.join(plot_dir, seq + "-err.txt")
            print("Saving errs to {}...".format(plot_err_file))
            np.savetxt(plot_err_file, np.array(errs))
            plot_loc_stats(
                seq, plot_dir, T_pred_seq, T_gt_seq, errs, consist, Xi, Cov, has_cov
            )
        rmse = root_mean_square(errs)
        seq_rmse.append(rmse)
        print(
            "RMSE: x: {} m y: {} m z: {} m roll: {} deg pitch: {} deg yaw: {} deg".format(
                rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]
            )
        )
        c = -1
        if has_cov:
            c = np.sqrt(max(0, np.mean(consist) / 6.0))
            # c = np.mean(np.sqrt(np.array(consist) / 6.0))
            print("Consistency: {}".format(c))
        seq_consist.append(c)
        print(" ")

    seq_rmse = np.array(seq_rmse)
    rmse = np.mean(seq_rmse, axis=0).squeeze()
    print(
        "Overall RMSE: x: {} m y: {} m z: {} m roll: {} deg pitch: {} deg yaw: {} deg".format(
            rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]
        )
    )
    c = -1
    if seqs_have_cov:
        c = np.mean(seq_consist)
        print("Overall Consistency: {}".format(c))
        con = np.array(seq_consist).reshape(-1, 1)
        errs = np.concatenate((seq_rmse, con), -1)
    else:
        errs = seq_rmse

    return errs, gt_seqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, help="path to prediction files")
    parser.add_argument("--gt", type=str, help="path to groundtruth sequences")
    parser.add_argument(
        "--ref_seq",
        default=loc_reference,
        type=str,
        help="Which sequence to use as a reference",
    )
    parser.add_argument(
        "--ref_sensor",
        default="lidar",
        type=str,
        help="Which sensor to use as a reference (camera|lidar|radar|aeva)",
    )
    parser.add_argument(
        "--test_sensor",
        default="lidar",
        type=str,
        help="Which sensor to use as the test sensor (camera|lidar|radar|aeva)",
    )
    parser.add_argument("--dim", default=3, type=int, help="SE(3) or SE(2)")
    parser.add_argument("--plot", type=str, help="path to save plots")
    args = parser.parse_args()
    assert args.ref_sensor in ["camera", "lidar", "radar", "aeva"]
    assert args.test_sensor in ["camera", "lidar", "radar", "aeva"]
    assert args.dim in [2, 3]
    if args.ref_sensor == "radar" or args.test_sensor == "radar":
        assert args.dim == 2
    os.makedirs(args.plot, exist_ok=True)
    eval_local(
        args.pred,
        args.gt,
        args.ref_seq,
        args.ref_sensor,
        args.test_sensor,
        args.dim,
        args.plot,
    )
