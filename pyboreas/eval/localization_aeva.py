import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyboreas.data.splits import loc_reference
from pyboreas.utils.odometry import plot_loc_stats, read_traj_file2, read_traj_file_gt2, get_sequence_velocities_gt
from pyboreas.utils.utils import (
    SE3Tose3,
    get_closest_index,
    get_inverse_tf,
    rotToRollPitchYaw,
)
import csv

"""
Evaluate aeva localization results including thresholding tests.
"""

def get_Tas(gtpath, seq, sensor="aeva"):
    T_applanix_aeva = np.loadtxt(osp.join(gtpath, seq, 'calib', 'T_applanix_aeva.txt'))
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

def take_mean(errs):
    return np.mean(np.array(errs), axis=0).squeeze()

# QOL: make gt data the same length as pred data
def adjust_length(gt_poses, gt_times, target_len):
    if len(gt_poses) > target_len:
        gt_poses = gt_poses[:target_len]
        gt_times = gt_times[:target_len]
    else:
        raise Exception("gt_poses is shorter than pred_poses")
    return gt_poses, gt_times

def plot_loc_with_rmse(T_pred, T_gt, rmse, save_loc_name, plot_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 9))
    # xy
    ax[0].plot(T_pred[:, 0], T_pred[:, 1], label='pred')
    ax[0].plot(T_gt[:, 0], T_gt[:, 1], linestyle='dashed', label = 'gt')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')
    ax[0].axis('equal')

    # xz
    ax[1].plot(T_pred[:, 0], T_pred[:, 2], label='pred')
    ax[1].plot(T_gt[:, 0], T_gt[:, 2], linestyle='dashed', label = 'gt')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('x [m]')
    ax[1].set_ylabel('z [m]')
    ax[1].axis('equal')

    # yz
    ax[2].plot(T_pred[:, 1], T_pred[:, 2], label='pred')
    ax[2].plot(T_gt[:, 1], T_gt[:, 2], linestyle='dashed', label = 'gt')
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlabel('y [m]')
    ax[2].set_ylabel('z [m]')
    ax[2].axis('equal')   

    fig.suptitle("RMSE: x: {} m y: {} m z: {} m\nroll: {} deg pitch: {} deg yaw: {} deg".format(
        rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]))
    fig.tight_layout()
    save_path = os.path.join(plot_dir, save_loc_name + '_path.png')
    print('Path saved to ', save_path)
    plt.savefig(save_path)
    plt.close()

def plot_3d_loc(T_pred, T_gt, save_loc_name, plot_dir):
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(T_pred[:, 0], T_pred[:, 1], T_pred[:, 2], label='pred')
    ax.plot(T_gt[:, 0], T_gt[:, 1], T_gt[:, 2], linestyle='dashed', label='gt')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid()
    ax.set_title('3D Trajectory')
    save_path_3d = os.path.join(plot_dir, save_loc_name + '_path_3D.png')
    print('3D Path saved to ', save_path_3d)
    plt.savefig(save_path_3d)

def plot_errs_with_time(errs, timestamps, save_loc_name, plot_dir):
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        
    ax[0, 0].plot(timestamps, errs[:, 0], label='x error')
    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('Error [m]')
    ax[0, 0].set_title('X Error')
    ax[0, 0].grid()
    
    ax[0, 1].plot(timestamps, errs[:, 1], label='y error')
    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel('Error [m]')
    ax[0, 1].set_title('Y Error')
    ax[0, 1].grid()
    
    ax[0, 2].plot(timestamps, errs[:, 2], label='z error')
    ax[0, 2].set_xlabel('Time [s]')
    ax[0, 2].set_ylabel('Error [m]')
    ax[0, 2].set_title('Z Error')
    ax[0, 2].grid()
    
    ax[1, 0].plot(timestamps, errs[:, 3], label='roll error')
    ax[1, 0].set_xlabel('Time [s]')
    ax[1, 0].set_ylabel('Error [deg]')
    ax[1, 0].set_title('Roll Error')
    ax[1, 0].grid()
    
    ax[1, 1].plot(timestamps, errs[:, 4], label='pitch error')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Error [deg]')
    ax[1, 1].set_title('Pitch Error')
    ax[1, 1].grid()
    
    ax[1, 2].plot(timestamps, errs[:, 5], label='yaw error')
    ax[1, 2].set_xlabel('Time [s]')
    ax[1, 2].set_ylabel('Error [deg]')
    ax[1, 2].set_title('Yaw Error')
    ax[1, 2].grid()
    
    fig.suptitle('Errors as a function of timestamps')
    fig.tight_layout()
    for i in range(2):
        for j in range(3):
            mean = np.mean(errs[:, i * 3 + j])
            stddev = np.std(errs[:, i * 3 + j])
            ax[i, j].legend(title=f"mean: {mean:.2f}, stddev: {stddev:.2f}")
    save_path_errors = os.path.join(plot_dir, save_loc_name + '_errors.png')
    plt.savefig(save_path_errors)
    plt.close()

def eval_boreas_local(
    predpath,
    gtpath,
    gt_ref_seq,
    ref_sensor="aeva",
    test_sensor="aeva",
    dim=3,
    plot_dir=None,
    loc_dir=None):
    
    pred_files = sorted(
        [
            f
            for f in os.listdir(predpath)
            if f.startswith("boreas") and f.endswith(".txt") and "err" not in f 
        ]
    )

    loc_name = sorted(
        [
            f.split('_threshold_')[0] + '.txt' if '_threshold_' in f else f
            for f in os.listdir(predpath)
            if f.startswith("boreas") and f.endswith(".txt") and "err" not in f
        ]
    )

    if loc_dir is not None:
        pred_files = [loc_dir + '.txt']
        loc_name = [loc_dir.split('_threshold_')[0] if '_threshold_' in loc_dir else loc_dir]

    gt_seqs = []
    for predfile in loc_name:
        if Path(predfile).stem.split(".")[0] not in os.listdir(gtpath):
            raise Exception(
                f"prediction file {predfile} doesn't match ground truth sequence list"
            )
        gt_seqs.append(Path(predfile).stem.split(".")[0])

    gt_ref_poses, gt_ref_times = read_traj_file_gt2(
        osp.join(gtpath, gt_ref_seq, "applanix", ref_sensor + "_poses.csv"), dim=dim
    )

    # get corresponding groundtruth poses
    vel_gt, vel_times, seq_lens_gt, crop = get_sequence_velocities_gt(gtpath, gt_seqs, dim)

    print(osp.join(gtpath, gt_seqs[0]))
    print(len(vel_gt))
    print(vel_gt[0])

    seq_rmse = []
    seq_consist = []
    seqs_have_cov = True
    for predfile, seq in zip(pred_files, gt_seqs):
        print("Processing {}...".format(seq))
        save_loc_name = predfile.replace('.txt', '')

        T_as = get_Tas(gtpath, seq, ref_sensor) # T_applanix_sensor
        T_sa = get_inverse_tf(T_as)             # T_sensor_applanix
        T_ax_app = np.array([[ 0.0299955,  0.99955003,  0, 0.51],
                             [-0.99955003,  0.0299955,  0, 0.0],
                             [ 0, 0, 1, 1.45],
                             [ 0, 0, 0, 1]]).astype(np.float64) # known extrinsic between applanix and vehicle
        T_rs = T_ax_app @ T_as                  # T_rear_axle_sensor
        print("T_rs: ", T_rs)
        T_sr = get_inverse_tf(T_rs)             # T_sensor_rear_axle
        print("T_sr: ", T_sr)
        pred_poses, pred_times, ref_times, cov_matrices, has_cov = read_traj_file2(
            osp.join(predpath, predfile)
        )
        seqs_have_cov *= has_cov
        gt_poses, gt_times = read_traj_file_gt2(
            osp.join(gtpath, seq, "applanix", test_sensor + "_poses.csv"), dim=dim
        )
        
        cutoff = len(pred_poses)
        print(len(gt_poses), len(pred_poses))

        # Iterate through gt_times and pred_times, remove missing timestamps in gt_times
        i = 0
        while i < len(pred_times):
            if pred_times[i] != gt_times[i]:
                print("Removing timestamp: ", gt_times[i])
                gt_times = np.delete(gt_times, i)
                gt_poses = np.delete(gt_poses, i, axis=0)
            else:
                i += 1
       
        # adjust the length of T_gt and seq_lens_gt
        gt_poses, gt_times = adjust_length(gt_poses, gt_times, cutoff) 

        # Compare pred times to gt times and list the indices where they differ
        differing_indices = np.where(np.array(pred_times) != np.array(gt_times))[0]
        if len(differing_indices) > 0:
            print("Indices where pred times differ from gt times:")
            print(differing_indices, len(differing_indices))
        else:
            print("No differences between pred times and gt times.")
        
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
            plot_err_file = osp.join(plot_dir, save_loc_name + "_err.txt")
            print("Saving errs to {}...".format(plot_err_file))
            np.savetxt(plot_err_file, np.array(errs))
            plot_loc_stats(
                save_loc_name, plot_dir, T_pred_seq, pred_times, vel_gt, vel_times, T_gt_seq, errs, consist, Xi, Cov, has_cov
            )
        rmse = root_mean_square(errs)
        seq_rmse.append(rmse)
        
        T_pred = np.array(
            [np.linalg.inv(T_i_vk)[:3, 3] for T_i_vk in T_pred_seq], dtype=np.float64
        )
        T_gt = np.array(
            [np.linalg.inv(T_i_vk)[:3, 3] for T_i_vk in T_gt_seq], dtype=np.float64
        )
        
        plot_loc_with_rmse(T_pred, T_gt, rmse, save_loc_name, plot_dir)
        plot_3d_loc(T_pred, T_gt, save_loc_name, plot_dir)

        if 'threshold' in save_loc_name:
            print("\033[91mSequence: {}\033[0m \033[91mThreshold: {}\033[0m".format(save_loc_name.split('_threshold_')[0], save_loc_name.split('_threshold_')[1]))

            log_dir = os.path.join(os.path.dirname(predpath), save_loc_name)
            
            # Get the last .log file in the directory
            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')])
            if log_files:
                last_log_file = log_files[-1]
                log_file_path = osp.join(log_dir, last_log_file)
                
                # third last line has time
                with open(log_file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 3:
                        print(lines[-3].strip().split("[tactic.module]")[-1])
                    else:
                        print("Log file has less than 3 lines.")

                    type = "icp"
                    for line in lines[-22:]:
                        if "odometry_doppler" in line:
                            type = "doppler"
                            break

            csv_file = os.path.join(plot_dir, '../../..', 'results.csv')
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = ['seq', 'type', 'thresh', 'time', 'x_rmse', 'y_rmse', 'z_rmse', 'roll_rmse', 'pitch_rmse', 'yaw_rmse']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'seq': save_loc_name.split('_threshold_')[0],
                    'type': type,
                    'thresh': save_loc_name.split('_threshold_')[1],
                    'time': lines[-3].strip().split("per frame: ")[-1].split(" ms")[0],
                    'x_rmse': rmse[0],
                    'y_rmse': rmse[1],
                    'z_rmse': rmse[2],
                    'roll_rmse': rmse[3],
                    'pitch_rmse': rmse[4],
                    'yaw_rmse': rmse[5]
                })

        print("\033[91mRMSE: x: {} m y: {} m z: {} m roll: {} deg pitch: {} deg yaw: {} deg\033[0m".format(
            rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]
        ))
           
        # Plot errors as a function of timestamps
        timestamps = (np.array(pred_times) - pred_times[0]) / 1e6
        errs = np.array(errs)
        plot_errs_with_time(errs, timestamps, save_loc_name, plot_dir)
        
        mean_err = take_mean(errs)
        print(
            "\033[91mmean: x: {} m y: {} m z: {} m roll: {} deg pitch: {} deg yaw: {} deg\033[0m".format(
            mean_err[0], mean_err[1], mean_err[2], mean_err[3], mean_err[4], mean_err[5]
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
        "\033[94mOverall RMSE: x: {} m y: {} m z: {} m roll: {} deg pitch: {} deg yaw: {} deg\033[0m".format(
            rmse[0], rmse[1], rmse[2], rmse[3], rmse[4], rmse[5]
        )
    )
    print("")
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
        default="aeva",
        type=str,
    )
    parser.add_argument(
        "--test_sensor",
        default="aeva",
        type=str,
    )
    parser.add_argument("--dim", default=3, type=int, help="SE(3) or SE(2)")
    parser.add_argument("--plot", type=str, help="path to save plots")
    parser.add_argument("--loc_dir", type=str, help="select a loc sequence")
    args = parser.parse_args()
    assert args.ref_sensor in ["aeva"]
    assert args.test_sensor in ["aeva"]
    assert args.dim in [2, 3]
    os.makedirs(args.plot, exist_ok=True)
    eval_boreas_local(
        args.pred,
        args.gt,
        args.ref_seq,
        args.ref_sensor,
        args.test_sensor,
        args.dim,
        args.plot,
        args.loc_dir
    )