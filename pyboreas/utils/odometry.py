import os
import os.path as osp
from itertools import accumulate, repeat
from multiprocessing import Pool
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from pylgmath import Transformation, se3op
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.vspace import VSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.solver import GaussNewtonSolver
from pysteam.trajectory import Time
from pysteam.trajectory.const_vel import Interface as TrajectoryInterface

from pyboreas.utils.utils import (
    enforce_orthog,
    get_inverse_tf,
    rotation_error,
    translation_error,
    yawPitchRollToRot,
)


class TrajStateVar:
    """This class defines a trajectory state variable for steam."""

    def __init__(
        self,
        time: Time,
        pose: SE3StateVar,
        velocity: VSpaceStateVar,
    ) -> None:
        self.time: Time = time
        self.pose: SE3StateVar = pose
        self.velocity: VSpaceStateVar = velocity


def interpolate_poses(poses, times, query_times, solver=True, verbose=False):
    """Runs a steam optimization with locked poses and outputs poses queried at query_times
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames)
        times (List[int]): list of times for poses (int for microseconds)
        query_times (List[int]): list of query times (int for microseconds)
        solver (bool): 'True' solves velocities with batch optimization. 'False' we use a finite-diff. approx.
        verbose (bool): verbose setting for steam solver
    Returns:
        (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames) at query_times
    """

    # WNOA Qc diagonal
    # Note: applanix frame is x-right, y-forward, z-up
    qcd = np.array([0.1, 1.0, 0.1, 0.01, 0.01, 0.1])

    # steam state variables
    states = []
    for i in range(len(poses)):
        if i == 0:
            dt = (times[1] - times[0]) * 1e-6  # microseconds to seconds
            dT = Transformation(T_ba=poses[1]) @ Transformation(T_ba=poses[0]).inverse()
        else:
            dt = (times[i] - times[i - 1]) * 1e-6  # microseconds to seconds
            dT = (
                Transformation(T_ba=poses[i])
                @ Transformation(T_ba=poses[i - 1]).inverse()
            )
        velocity = dT.vec() / dt  # initializing with finite difference
        states += [
            TrajStateVar(
                Time(nsecs=int(times[i] * 1e3)),  # microseconds to nano
                SE3StateVar(Transformation(T_ba=poses[i]), locked=True),
                VSpaceStateVar(velocity),
            )
        ]

    # setup trajectory
    traj = TrajectoryInterface(qcd)
    for state in states:
        traj.add_knot(time=state.time, T_k0=state.pose, w_0k_ink=state.velocity)

    if solver:
        # construct the optimization problem
        opt_prob = OptimizationProblem()
        opt_prob.add_cost_term(*traj.get_prior_cost_terms())
        opt_prob.add_state_var(*[j for i in states for j in (i.pose, i.velocity)])

        # construct the solver
        optimizer = GaussNewtonSolver(opt_prob, verbose=verbose, use_sparse_matrix=True)

        # solve the problem (states are automatically updated)
        optimizer.optimize()

    query_poses = []
    for time_us in query_times:
        interp_eval = traj.get_pose_interpolator(Time(nsecs=int(time_us * 1e3)))
        query_poses += [enforce_orthog(interp_eval.evaluate().matrix())]

    return query_poses


def trajectory_distances(poses):
    """Calculates path length along the trajectory.
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_k_i, 'i' is a fixed reference frame)
    Returns:
        List[float]: distance along the trajectory, increasing as a function of time / list index
    """
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i - 1] + np.sqrt(dx**2 + dy**2 + dz**2))
    return dist


def last_frame_from_segment_length(dist, first_frame, length):
    """Retrieves the index of the last frame for our current analysis.
        last_frame should be 'dist' meters away from first_frame in terms of distance traveled along the trajectory.
    Args:
        dist (List[float]): distance along the trajectory, increasing as a function of time / list index
        first_frame (int): index of the starting frame for this sequence
        length (float): length of the current segment being evaluated
    Returns:
        last_frame (int): index of the last frame in this segment
    """
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def calc_sequence_errors(poses_gt, poses_pred, step_size, dim=3):
    """Calculate the translation and rotation error for each subsequence across several different lengths.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix, ground truth transforms
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix, predicted transforms
        step_size (int): step size applied for computing distances travelled
    Returns:
        err (List[Tuple]): each entry in list is [first_frame, r_err, t_err, length, speed]
        lengths (List[int]): list of lengths that odometry is evaluated at
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    # Pre-compute distances from ground truth as reference
    dist = trajectory_distances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(
                poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame])
            )
            pose_delta_res = np.matmul(
                poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame])
            )
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            if dim == 2:
                pose_error_vec = se3op.tran2vec(pose_error)  # T_gt_pred
                pose_error_vec[2:5] = 0  # z and roll, pitch to 0
                pose_error = se3op.vec2tran(pose_error_vec)
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.1 * num_frames)
            err.append(
                [
                    first_frame,
                    r_err / float(length),
                    t_err / float(length),
                    length,
                    speed,
                ]
            )
    return err, lengths


def get_stats(err, lengths):
    """Computes the average translation and rotation within a sequence (across subsequences of diff lengths).
    Args:
        err (List[Tuple]): each entry in list is [first_frame, r_err, t_err, length, speed]
        lengths (List[int]): list of lengths that odometry is evaluated at
    Returns:
        average translation (%) and rotation (deg/m) errors
    """
    t_err = 0
    r_err = 0
    len2id = {x: i for i, x in enumerate(lengths)}
    t_err_len = [0.0] * len(len2id)
    r_err_len = [0.0] * len(len2id)
    len_count = [0] * len(len2id)
    for e in err:
        t_err += e[2]
        r_err += e[1]
        j = len2id[e[3]]
        t_err_len[j] += e[2]
        r_err_len[j] += e[1]
        len_count[j] += 1
    t_err /= float(len(err))
    r_err /= float(len(err))
    return (
        t_err * 100,
        r_err * 180 / np.pi,
        [a / float(b) * 100 for a, b in zip(t_err_len, len_count)],
        [a / float(b) * 180 / np.pi for a, b in zip(r_err_len, len_count)],
    )

def plot_stats(seq, dir, T_odom, T_gt, lengths, t_err_len, r_err_len, t_err=None, r_err=None, err_2d_per_frame=None, err_stats_2d=None):
    """Outputs plots of calculated statistics to specified directory.
    Args:
        seq (List[string]): list of sequence file names
        dir (string): directory path for plot outputs
        T_odom (List[np.ndarray]): list of 4x4 estimated poses T_vk_i (vehicle frame at time k and fixed frame i)
        T_gt (List[np.ndarray]): List of 4x4 groundtruth poses T_vk_i (vehicle frame at time k and fixed frame i)
        lengths (List[int]): list of lengths that odometry is evaluated at
        t_err_len (List[float]): list of average translation error corresponding to lengths
        r_err_len (List[float]): list of average rotation error corresponding to lengths
        t_err (float): translation error
        r_err (float): rotation error
        err_2d_per_frame (Dict): dictionary of average 2D translation and rotation errors per frame
        err_stats_2d (Dict): is a dictionary with key as the first frame of the sequence and value as a list of [r_err, t_err, count, err_per_length]
    """
    path_odom, path_gt = get_path_from_Tvi_list(T_odom, T_gt)

    # plot of path (xy view)
    plt.figure(figsize=(6, 6))
    plt.plot(path_odom[:, 0], path_odom[:, 1], "b", linewidth=0.5, label="Estimate")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    if t_err is not None and r_err is not None:
        plt.title(f"Path\nTranslation Error: {t_err:.2f}% | Rotation Error: {r_err*100:.4f} deg/100m")
    else:
        plt.title("Path")
    plt.savefig(
        osp.join(dir, seq[:-4] + "_path.pdf"), pad_inches=0, bbox_inches="tight"
    )
    plt.close()

    # plot of path (xz view)
    plt.figure(figsize=(6, 6))
    plt.plot(path_odom[:, 0], path_odom[:, 2], "b", linewidth=0.5, label="Estimate")
    plt.plot(path_gt[:, 0], path_gt[:, 2], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 2],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(dir, seq[:-4] + "_path_xz.pdf"), pad_inches=0, bbox_inches="tight"
    )
    plt.close()

    # plot of translation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, t_err_len, "bs", markerfacecolor="none")
    plt.plot(lengths, t_err_len, "b")
    plt.xlabel("Path Length [m]")
    plt.ylabel("Translation Error [%]")
    plt.savefig(
        osp.join(dir, seq[:-4] + "_tl.pdf"), pad_inches=0, bbox_inches="tight"
    )
    plt.close()

    # plot of rotation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, r_err_len, "bs", markerfacecolor="none")
    plt.plot(lengths, r_err_len, "b")
    plt.xlabel("Path Length [m]")
    plt.ylabel("Rotation Error [deg/m]")
    plt.savefig(
        osp.join(dir, seq[:-4] + "_rl.pdf"), pad_inches=0, bbox_inches="tight"
    )
    plt.close()

    if err_2d_per_frame is not None:
        err_2d_per_frame = dict(sorted(err_2d_per_frame.items()))
        path_odom = path_odom[list(err_2d_per_frame.keys())]
        path_gt = path_gt[list(err_2d_per_frame.keys())]
        # plot the gt trajectory and the estimated trajectory, but the estimated trajectory is colored by the rotation error
        plt.figure(figsize=(6, 6))
        plt.plot(path_odom[:, 0], path_odom[:, 1], "b", linewidth=0.5, label="Estimate")
        plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
        plt.plot(
            path_gt[0, 0],
            path_gt[0, 1],
            "ks",
            markerfacecolor="none",
            label="Sequence Start",
        )
        sc = plt.scatter(
            path_odom[:len(err_2d_per_frame.keys()), 0],
            path_odom[:len(err_2d_per_frame.keys()), 1],
            c=[r_err_len for r_err_len, _ in err_2d_per_frame.values()],
            cmap="viridis",
            label="Rotation Error",
            s=5  # reduce the size of points
        )
        plt.colorbar(sc, label="Rotation Error [deg/m]")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.legend(loc="upper right")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_path_r_err.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()

        # plot the gt trajectory and the estimated trajectory, but the estimated trajectory is colored by the translation error
        plt.figure(figsize=(6, 6))
        plt.plot(path_odom[:, 0], path_odom[:, 1], "b", linewidth=0.5, label="Estimate")
        plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
        plt.plot(
            path_gt[0, 0],
            path_gt[0, 1],
            "ks",
            markerfacecolor="none",
            label="Sequence Start",
        )
        sc = plt.scatter(
            path_odom[:len(err_2d_per_frame.keys()), 0],
            path_odom[:len(err_2d_per_frame.keys()), 1],
            c=[t_err_len for _, t_err_len in err_2d_per_frame.values()],
            cmap="viridis",
            label="Translation Error",
            s=5 # reduce the size of points
        )
        plt.colorbar(sc, label="Translation Error [%]")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.legend(loc="upper right")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_path_t_err.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()

        # plot of translation error per frame
        plt.figure(figsize=(6, 3))
        # plt.plot(err_2d_per_frame.keys(), [v[1] for v in err_2d_per_frame.values()], "bs", markerfacecolor="none")
        plt.plot(err_2d_per_frame.keys(), [v[1] for v in err_2d_per_frame.values()], "b")
        plt.xlabel("Frame")
        plt.ylabel("Translation Error [%]")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_tl_frame.pdf"), pad_inches=0, bbox_inches="tight"
        )
        plt.close()

        # plot of rotation error per frame
        plt.figure(figsize=(6, 3))
        # plt.plot(err_2d_per_frame.keys(), [v[0] for v in err_2d_per_frame.values()], "bs", markerfacecolor="none")
        plt.plot(err_2d_per_frame.keys(), [v[0] for v in err_2d_per_frame.values()], "b")
        plt.xlabel("Frame")
        plt.ylabel("Rotation Error [deg/m]")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_rl_frame.pdf"), pad_inches=0, bbox_inches="tight"
        )
        plt.close()

    if err_stats_2d is not None:
        # err_stats_2d is a dictionary with key as the first frame of the sequence and value as a list of [r_err, t_err, count, err_per_length]
        # plot rotational error per length
        plt.figure(figsize=(10, 5))
        for i, length in enumerate(lengths):
            plt.plot(
                list(err_stats_2d.keys()),
                [v[3][i] for v in err_stats_2d.values()],
                label=f"Rotational Error {length}m",
            )
        plt.xlabel("Frame")
        plt.ylabel("Rotational Error")
        plt.legend(loc="upper right")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_rotational_error_per_length.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()

        # plot translational error per length
        plt.figure(figsize=(10, 5))
        for i, length in enumerate(lengths):
            plt.plot(
                list(err_stats_2d.keys()),
                [v[3][len(lengths) + i] for v in err_stats_2d.values()],
                label=f"Translational Error {length}m",
            )
        plt.xlabel("Frame")
        plt.ylabel("Translational Error")
        plt.legend(loc="upper right")
        plt.savefig(
            osp.join(dir, seq[:-4] + "_translational_error_per_length.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()
    


def plot_loc_stats(
    seq, plot_dir, T_loc, T_gt, errs, consist=[], Xi=[], Cov=[], has_cov=False
):
    print(f"Plotting localization results for {seq}...")
    path_loc = np.array(
        [np.linalg.inv(T_i_vk)[:3, 3] for T_i_vk in T_loc], dtype=np.float64
    )
    path_gt = np.array(
        [np.linalg.inv(T_i_vk)[:3, 3] for T_i_vk in T_gt], dtype=np.float64
    )
    # plot of path
    plt.figure(figsize=(6, 6))
    plt.plot(path_loc[:, 0], path_loc[:, 1], "b", linewidth=0.5, label="Estimate")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_path.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

    # plot of errors vs. time
    if len(Xi) > 0 and len(Cov) > 0:
        # plt.rcParams.update({"text.usetex": True})
        fig, axs = plt.subplots(6, 1, figsize=(6, 12))
        Sigma = 3 * np.sqrt(Cov)
        axs[0].plot(Xi[:, 0], color="limegreen", linewidth=1)
        axs[0].plot(Sigma[:, 0], color="k", linewidth=1)
        axs[0].plot(-Sigma[:, 0], color="k", linewidth=1)
        axs[0].set_ylabel("rho_1")

        axs[1].plot(Xi[:, 1], color="limegreen", linewidth=1)
        axs[1].plot(Sigma[:, 1], color="k", linewidth=1)
        axs[1].plot(-Sigma[:, 1], color="k", linewidth=1)
        axs[1].set_ylabel("rho_2")

        axs[2].plot(Xi[:, 2], color="limegreen", linewidth=1)
        axs[2].plot(Sigma[:, 2], color="k", linewidth=1)
        axs[2].plot(-Sigma[:, 2], color="k", linewidth=1)
        axs[2].set_ylabel("rho_3")

        axs[3].plot(Xi[:, 3], color="limegreen", linewidth=1)
        axs[3].plot(Sigma[:, 3], color="k", linewidth=1)
        axs[3].plot(-Sigma[:, 3], color="k", linewidth=1)
        axs[3].set_ylabel("psi_1")

        axs[4].plot(Xi[:, 4], color="limegreen", linewidth=1)
        axs[4].plot(Sigma[:, 4], color="k", linewidth=1)
        axs[4].plot(-Sigma[:, 4], color="k", linewidth=1)
        axs[4].set_ylabel("psi_2")

        axs[5].plot(Xi[:, 5], color="limegreen", linewidth=1)
        axs[5].plot(Sigma[:, 5], color="k", linewidth=1)
        axs[5].plot(-Sigma[:, 5], color="k", linewidth=1)
        axs[5].set_ylabel("psi_3")
        axs[5].set_xlabel("time (s)")
        plt.savefig(
            osp.join(plot_dir, seq + "_errs.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()

        e = np.array(errs)
        fig, axs = plt.subplots(2, 2, figsize=(8, 7))
        axs[0, 0].hist(e[:, 0], bins=20)
        axs[0, 0].set_title("Lateral Error (m)")
        axs[0, 1].hist(e[:, 1], bins=20)
        axs[0, 1].set_title("Longitudinal Error (m)")
        axs[1, 0].hist(e[:, 2], bins=20)
        axs[1, 0].set_title("Vertical Error (m)")
        axs[1, 1].hist(e[:, 3], bins=20)
        axs[1, 1].set_title("Orientation Error (deg)")
        plt.savefig(
            osp.join(plot_dir, seq + "_hist.pdf"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close()

    e = np.array(errs)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0,0].hist(e[:, 0], bins=20)
    axs[0, 0].set_title("Lateral Error (m)")
    axs[0, 0].legend([f"median: {np.median(e[:, 0]):.2f} m"], loc="upper right")
    axs[0, 1].hist(e[:, 1], bins=20)
    axs[0, 1].set_title("Longitudinal Error (m)")
    axs[0, 1].legend([f"median: {np.median(e[:, 1]):.2f} m"], loc="upper right")
    axs[0, 2].hist(e[:, 2], bins=20)
    axs[0, 2].set_title("Vertical Error (m)")
    axs[0, 2].legend([f"median: {np.median(e[:, 2]):.2f} m"], loc="upper right")
    axs[1, 0].hist(e[:, 3], bins=20)
    axs[1, 0].set_title("Roll Error (deg)")
    axs[1, 0].legend([f"median: {np.median(e[:, 3]):.2f} deg"], loc="upper right")
    axs[1, 1].hist(e[:, 4], bins=20)
    axs[1, 1].set_title("Pitch Error (deg)")
    axs[1, 1].legend([f"median: {np.median(e[:, 4]):.2f} deg"], loc="upper right")
    axs[1, 2].hist(e[:, 5], bins=20)
    axs[1, 2].set_title("Yaw Error (deg)")
    axs[1, 2].legend([f"median: {np.median(e[:, 5]):.2f} deg"], loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_hist.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

    # Calculate RMSE
    rmse = root_mean_square(errs)
    rmse_x = rmse[0]
    rmse_y = rmse[1]

    # Plot path colored by x RMSE
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        path_loc[:, 0],
        path_loc[:, 1],
        c=np.abs(e[:, 0]),
        cmap="viridis",
        label=f"X RMSE: {rmse_x:.2f} m",
        s=5  # reduce the size of points
    )
    plt.colorbar(sc, label="X Error [m]")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_path_x_rmse.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

    # Plot path colored by y RMSE
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        path_loc[:, 0],
        path_loc[:, 1],
        c=np.abs(e[:, 1]),
        cmap="viridis",
        label=f"Y RMSE: {rmse_y:.2f} m",
        s=5  # reduce the size of points
    )
    plt.colorbar(sc, label="Y Error [m]")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_path_y_rmse.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

    # Plot path colored by z RMSE
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        path_loc[:, 0],
        path_loc[:, 1],
        c=np.abs(e[:, 2]),
        cmap="viridis",
        label=f"Z RMSE: {rmse[2]:.2f} m",
        s=5  # reduce the size of points
    )
    plt.colorbar(sc, label="Z Error [m]")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_path_z_rmse.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

    # Plot path colored by yaw RMSE
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        path_loc[:, 0],
        path_loc[:, 1],
        c=np.abs(e[:, 5]),
        cmap="viridis",
        label=f"Yaw RMSE: {rmse[5]:.2f} deg",
        s=5  # reduce the size of points
    )
    plt.colorbar(sc, label="Yaw Error [deg]")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "--r", linewidth=0.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.savefig(
        osp.join(plot_dir, seq + "_path_yaw_rmse.pdf"),
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()

def root_mean_square(errs):
    return np.sqrt(np.mean(np.power(np.array(errs), 2), axis=0)).squeeze()

def get_path_from_Tvi_list(T_vi_odom, T_vi_gt):
    """Gets 3D path (xyz) from list of poses T_vk_i (transform between vehicle frame at time k and fixed frame i) and
    aligns the groundtruth path with the estimated path.
    Args:
        T_vi_odom (List[np.ndarray]): list of 4x4 estimated poses T_vk_i (vehicle frame at time k and fixed frame i)
        T_vi_gt (List[np.ndarray]): List of 4x4 groundtruth poses T_vk_i (vehicle frame at time k and fixed frame i)
    Returns:
        path_odom (np.ndarray): K x 3 numpy array of estimated xyz coordinates in (0'd position) groundtruth inertial frame
        path_gt (np.ndarray): K x 3 numpy array of groundtruth xyz coordinates in (0'd position) groundtruth inertial frame
    """
    assert len(T_vi_odom) == len(T_vi_gt)  # assume 1:1 correspondence
    T_iv_odom = [np.linalg.inv(T_vk_i_odom) for T_vk_i_odom in T_vi_odom]
    T_iv_gt = [np.linalg.inv(T_vk_i_gt) for T_vk_i_gt in T_vi_gt]

    # Zero out the position of the first pose in the groundtruth to make plotting nicer
    pose_0_gt = T_iv_gt[0].copy()
    pose_0_gt[:3, :3] = np.zeros(3)
    T_iv_gt = [T_iv_gt_i - pose_0_gt for T_iv_gt_i in T_iv_gt]

    # Align odometry estimate to groundtruth inertial frame
    T_gt_odom_i = T_iv_gt[0] @ np.linalg.inv(T_iv_odom[0])  # align the first pose
    T_iv_odom_aligned = [T_gt_odom_i @ T_i_vk_odom for T_i_vk_odom in T_iv_odom]

    path_odom = np.array([T_i_vk[:3, 3] for T_i_vk in T_iv_odom_aligned], dtype=np.float64)
    path_gt = np.array([T_i_vk[:3, 3] for T_i_vk in T_iv_gt], dtype=np.float64)

    return path_odom, path_gt


def compute_interpolation_one_seq(T_pred, times_gt, times_pred, out_fname, solver):
    """Interpolate for poses at the groundtruth times and write them out as txt files.
    Args:
        T_pred (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        times_gt (List[int]): List of times (microseconds) corresponding to T_gt
        times_pred (List[int]): List of times (microseconds) corresponding to T_pred
        out_fname (string): path to output file for interpolation output
        solver (bool): 'True' solves velocities for built-in interpolation. 'False' we use a finite-diff. approx.
    Returns:
        Nothing
    """
    T_query = interpolate_poses(T_pred, times_pred, times_gt, solver)  # interpolate
    write_traj_file(out_fname, T_query, times_gt)  # write out
    print(
        f"interpolated sequence {osp.basename(out_fname)}, output file: {out_fname}"
    )

    return


def compute_interpolation(
    T_pred,
    times_gt,
    times_pred,
    seq_lens_gt,
    seq_lens_pred,
    seq,
    out_dir,
    solver,
    processes,
):
    """Interpolate for poses at the groundtruth times and write them out as txt files.
    Args:
        T_pred (List[np.ndarray]): List of 4x4 SE(3) transforms (fixed reference frame 'i' to frame 'v', T_vi)
        times_gt (List[int]): List of times (microseconds) corresponding to T_gt
        times_pred (List[int]): List of times (microseconds) corresponding to T_pred
        seq_lens_gt (List[int]): List of sequence lengths corresponding to T_gt
        seq_lens_pred (List[int]): List of sequence lengths corresponding to T_pred
        seq (List[string]): List of sequence file names
        out_dir (string): path to output directory for interpolation output
        solver (bool): 'True' solves velocities for built-in interpolation. 'False' we use a finite-diff. approx.
    Returns:
        Nothing
    """
    # get start and end indices of each sequence
    indices_gt = tuple(accumulate(seq_lens_gt, initial=0))
    indices_pred = tuple(accumulate(seq_lens_pred, initial=0))

    # prepare input iterators to compute_interpolation_one_seq
    T_pred_seq = (
        T_pred[indices_pred[i] : indices_pred[i + 1]] for i in range(len(seq_lens_pred))
    )
    times_gt_seq = (
        times_gt[indices_gt[i] : indices_gt[i + 1]] for i in range(len(seq_lens_pred))
    )
    times_pred_seq = (
        times_pred[indices_pred[i] : indices_pred[i + 1]]
        for i in range(len(seq_lens_pred))
    )
    out_fname_seq = (osp.join(out_dir, seq[i]) for i in range(len(seq_lens_pred)))
    solver_seq = repeat(solver, len(seq_lens_pred))

    if processes == 1:
        # loop for each sequence
        for i in range(len(seq_lens_pred)):
            ts = time()  # start time

            # get poses and times of current sequence
            T_pred_i = next(T_pred_seq)
            times_gt_i = next(times_gt_seq)
            times_pred_i = next(times_pred_seq)

            # query predicted trajectory at groundtruth times and write out
            print("interpolating sequence", seq[i], "...")
            T_query = interpolate_poses(
                T_pred_i, times_pred_i, times_gt_i, solver
            )  # interpolate
            write_traj_file(
                osp.join(out_dir, seq[i]), T_query, times_gt_i
            )  # write out
            print(seq[i], "took", str(time() - ts), " seconds")
            print("output file:", osp.join(out_dir, seq[i]), "\n")
    else:
        # compute interpolation for each sequence in parallel
        with Pool(processes) as p:
            ts = time()  # start time

            print(
                f"interpolating {len(seq_lens_pred)} sequences in parallel using {processes} workers ..."
            )
            p.starmap(
                compute_interpolation_one_seq,
                zip(
                    T_pred_seq, times_gt_seq, times_pred_seq, out_fname_seq, solver_seq
                ),
            )
            print(f"interpolation took {time() - ts:.2f} seconds\n")

    return


def get_stats_per_frame(err, lengths) :
    """Computes the average translation and rotation within a sequence (across subsequences of diff lengths) per each frame.
    Args:
        err (List[Tuple]): each entry in list is [first_frame, r_err, t_err, length, speed]
        lengths (List[int]): list of lengths that odometry is evaluated at
    Returns:
        average translation (%) and rotation (deg/m) errors
    """

    err_dict = {}
    for e in err:
        first_frame = e[0]
        r_err = e[1]
        t_err = e[2]
        l_id = e[3]
        if first_frame not in err_dict:
            err_dict[first_frame] = [r_err, t_err, 1, np.zeros(len(lengths)*2)]
        else:
            err_dict[first_frame][0] += r_err
            err_dict[first_frame][1] += t_err
            err_dict[first_frame][2] += 1
        err_dict[first_frame][3][lengths.index(l_id)] = r_err *100
        err_dict[first_frame][3][len(lengths) + lengths.index(l_id)] = t_err *100

    avg_err_dict = {k: (v[0] / v[2] * 100, v[1] / v[2] * 100) for k, v in err_dict.items()}
    return avg_err_dict, err_dict


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
        if len(T_pred_seq) != len(T_gt_seq):
            T_pred_seq = T_pred_seq[crop[i][0] : crop[i][1]]

        # 2d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size, 2)
        t_err_2d, r_err_2d, _, _ = get_stats(err, path_lengths)

        err_2d_per_frame, err_stats_2d = get_stats_per_frame(err, path_lengths)

        # 3d
        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size)
        t_err, r_err, t_err_len, r_err_len = get_stats(err, path_lengths)

        err_3d_per_frame, err_stats_3d = get_stats_per_frame(err, path_lengths)

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
                t_err,
                r_err,
                err_2d_per_frame,
                err_stats_2d
            )

    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]

    return t_err, r_err, err_list


def get_sequences(path, file_ext=""):
    """Retrieves a list of all the sequences in the dataset with the given prefix.
    Args:
        path (string): directory path to where the files are
        file_ext (string): string identifier to look for (e.g., '.txt')
    Returns:
        sequences (List[string]): list of sequence file names
    """
    sequences = [f for f in os.listdir(path) if f.endswith(file_ext)]
    sequences.sort()
    return sequences


def get_sequence_poses(path, seq):
    """Retrieves a list of the poses corresponding to the given sequences in the given file path.
    Args:
        path (string): directory path to where the files are
        seq (List[string]): list of sequence file names
    Returns:
        all_poses (List[np.ndarray]): list of 4x4 poses from all sequence files
        all_times (List[int]): list of times in nanoseconds from all sequence files
        seq_lens (List[int]): list of sequence lengths
    """

    # loop for each sequence
    all_poses = []
    all_times = []
    seq_lens = []
    for filename in seq:
        # parse file for list of poses and times
        poses, times = read_traj_file(osp.join(path, filename))
        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)

    return all_poses, all_times, seq_lens


def get_sequence_poses_gt(path, seq, dim, aeva):
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
    crop = []
    for filename in seq:
        # determine path to gt file
        dir = filename[:-4]  # assumes last four characters are '.txt'
        if dim == 3:
            if aeva:
                print("Using AEVA poses for 3D evaluation")
                filepath = osp.join(
                    path, dir, "applanix/aeva_poses.csv"
                ) # use 'aeva_poses.csv' for groundtruth
                T_calib = np.loadtxt(osp.join(path, dir, "calib/T_applanix_aeva.txt"))
            else:
                filepath = osp.join(
                    path, dir, "applanix/lidar_poses.csv"
                )  # use 'lidar_poses.csv' for groundtruth
                T_calib = np.loadtxt(osp.join(path, dir, "calib/T_applanix_lidar.txt"))
            poses, times = read_traj_file_gt(filepath, T_calib, dim)
            times_np = np.stack(times)

            filepath = osp.join(path, dir, 'applanix/camera_poses.csv')  # read in timestamps of camera groundtruth
            _, ctimes = read_traj_file_gt(filepath, np.identity(4), dim)
            if len(ctimes) == 0:
                # Something went wrong with loading camera timestamps, throw error
                raise ValueError(f"No camera timestamps found for sequence {dir}.")
            istart = np.searchsorted(times_np, ctimes[0])
            iend = np.searchsorted(times_np, ctimes[-1])
            poses = poses[istart:iend]
            times = times[istart:iend]
            crop += [(istart, iend)]

        elif dim == 2:
            filepath = osp.join(
                path, dir, "applanix/radar_poses.csv"
            )  # use 'radar_poses.csv' for groundtruth
            T_calib = np.identity(4)
            poses, times = read_traj_file_gt(filepath, T_calib, dim)
            crop += [(0, len(poses))]
        else:
            raise ValueError(
                "Invalid dim value in get_sequence_poses_gt. Use either 2 or 3."
            )

        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)
    return all_poses, all_times, seq_lens, crop


def get_sequence_times_gt(path, seq):
    """Retrieves a list of groundtruth (lidar) timestamps corresponding to the given sequences for 3D evaluation
    Args:
        path (string): directory path to root directory of Boreas dataset
        seq (List[string]): list of sequence file names
    Returns:
        all_times (List[int]): list of times in microseconds from all sequence files
        seq_lens (List[int]): list of sequence lengths
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    """
    # loop for each sequence
    all_times = []
    seq_lens = []
    crop = []
    for filename in seq:
        # determine path to gt file
        dir = filename[:-4]  # assumes last four characters are '.txt'
        lfilepath = osp.join(
            path, dir, "applanix/lidar_poses.csv"
        )  # use 'lidar_poses.csv' for groundtruth
        cfilepath = osp.join(
            path, dir, "applanix/camera_poses.csv"
        )  # read in timestamps of camera groundtruth
        if osp.isfile(lfilepath) and osp.isfile(cfilepath):
            # csv files exist, use them
            _, times = read_traj_file_gt(lfilepath, np.identity(4), dim=3)
            times_np = np.stack(times)
            _, ctimes = read_traj_file_gt(cfilepath, np.identity(4), dim=3)
        else:
            # read timestamps from data
            lpath = osp.join(path, dir, "lidar")  # read lidar data filenames
            times = [int(Path(f).stem) for f in os.listdir(lpath) if ".bin" in f]
            times.sort()
            times_np = np.stack(times)

            cpath = osp.join(path, dir, "camera")  # read camera data filenames
            ctimes = [int(Path(f).stem) for f in os.listdir(cpath) if ".png" in f]
            ctimes.sort()

        istart = np.searchsorted(times_np, ctimes[0])
        iend = np.searchsorted(times_np, ctimes[-1])
        times = times[istart:iend]
        crop += [(istart, iend)]
        if times[0] < ctimes[0] or times[-1] > ctimes[-1]:
            raise ValueError("Invalid start and end indices for groundtruth.")

        seq_lens.append(len(times))
        all_times.extend(times)

    return all_times, seq_lens, crop


def write_traj_file(path, poses, times):
    """Writes trajectory into a space-separated txt file
    Args:
        path (string): file path including file name
        poses (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames)
        times (List[int]): list of times for poses
    """
    with open(path, "w") as file:
        # Writing each time (nanoseconds) and pose to file
        for ts, pose in zip(times, poses):
            line = [ts]
            line.extend(pose.reshape(16)[:12].tolist())
            file.write(" ".join(str(num) for num in line))
            file.write("\n")


def read_traj_file(path):
    """Reads trajectory from a space-separated txt file
    Args:
        path (string): file path including file name
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as file:
        # read each time and pose to lists
        poses = []
        times = []

        for line in file:
            line_split = line.strip().split()
            values = [float(v) for v in line_split[1:]]
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(enforce_orthog(pose))
            times.append(int(line_split[0]))

    return poses, times


def read_traj_file2(path):
    """Reads trajectory from a space-separated txt file
    Args:
        path (string): file path including file name
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as file:
        # read each time and pose to lists
        poses = []
        pred_times = []
        ref_times = []
        cov_matrices = []
        has_cov = True

        for line in file:
            line_split = line.strip().split()
            values = [float(v) for v in line_split[2:]]
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(enforce_orthog(pose))
            pred_times.append(int(line_split[0]))
            ref_times.append(int(line_split[1]))
            if not has_cov:
                continue
            if len(values) == 48:
                cov_matrix = np.array(values[12:]).reshape(6, 6)
            else:
                cov_matrix = np.identity(6)
                has_cov = False
            cov_matrices.append(cov_matrix)

    return poses, pred_times, ref_times, cov_matrices, has_cov


def read_traj_file_gt(path, T_ab, dim):
    """Reads trajectory from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses (from world to sensor frame)
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as f:
        lines = f.readlines()
    poses = []
    times = []

    T_ab = enforce_orthog(T_ab)
    for line in lines[1:]:
        pose, time = convert_line_to_pose(line, dim)
        poses += [
            enforce_orthog(T_ab @ get_inverse_tf(pose))
        ]  # convert T_iv to T_vi and apply calibration
        times += [int(time)]  # microseconds
    return poses, times


def read_traj_file_gt2(path, dim=3):
    """Reads trajectory from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as f:
        lines = f.readlines()
    poses = []
    times = []
    for line in lines[1:]:
        pose, time = convert_line_to_pose(line, dim)
        poses.append(pose)
        times.append(time)  # microseconds
    return poses, times


def convert_line_to_pose(line, dim=3):
    """Reads trajectory from list of strings (single row of the comma-separeted groundtruth file). See Boreas
    documentation for format
    Args:
        line (List[string]): list of strings
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (np.ndarray): 4x4 SE(3) pose
        (int): time in nanoseconds
    """
    # returns T_iv
    line = line.replace("\n", ",").split(",")
    line = [float(i) for i in line[:-1]]
    # x, y, z -> 1, 2, 3
    # roll, pitch, yaw -> 7, 8, 9
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = line[1]  # x
    T[1, 3] = line[2]  # y
    if dim == 3:
        T[2, 3] = line[3]  # z
        T[:3, :3] = yawPitchRollToRot(line[9], line[8], line[7])
    elif dim == 2:
        T[:3, :3] = yawPitchRollToRot(
            line[9],
            np.round(line[8] / np.pi) * np.pi,
            np.round(line[7] / np.pi) * np.pi,
        )
    else:
        raise ValueError(
            "Invalid dim value in convert_line_to_pose. Use either 2 or 3."
        )
    time = int(line[0])
    return T, time

def get_sequence_velocities(path, seq, dim):
    """Retrieves a list of the velocities corresponding to the given sequences in the given file path.
    Args:
        path (string): directory path to where the files are
        seq (List[string]): list of sequence file names
    Returns:
        all_velocities (List[np.ndarray]): list of 6x1 poses from all sequence files
        all_times (List[int]): list of times in nanoseconds from all sequence files
        seq_lens (List[int]): list of sequence lengths
    """

    # loop for each sequence
    all_velocities = []
    all_times = []
    seq_lens = []
    for filename in seq:
        # parse file for list of poses and times
        vel, times = read_vel_file(osp.join(path, filename), dim)
        seq_lens.append(len(times))
        all_velocities.extend(vel)
        all_times.extend(times)

    return all_velocities, all_times, seq_lens

def read_vel_file(path, dim=3):
    """Reads velocity from a space-separated txt file
    Args:
        path (string): file path including file name
    Returns:
        (List[np.ndarray]): list of 6x1 velocities
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as file:
        # read each time and pose to lists
        velocities = []
        times = []

        for line in file:
            line_split = line.strip().split()
            values = [float(v) for v in line_split[1:]]
            vel = np.zeros((6, 1), dtype=np.float64)
            vel[:, 0] = values
            # If 2D, zero out z, roll, pitch
            if dim == 2:
                vel[2:5] = 0.0

            velocities.append(vel)
            times.append(int(line_split[0]))

    return velocities, times

def get_sequence_velocities_gt(path, seq, dim, aeva=False):
    """Retrieves a list of the velocities corresponding to the given sequences in the given file path with the Boreas dataset
    directory structure.
    Args:
        path (string): directory path to root directory of Boreas dataset
        seq (List[string]): list of sequence file names
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        all_velocities (List[np.ndarray]): list of 4x4 poses from all sequence files
        all_times (List[int]): list of times in microseconds from all sequence files
        seq_lens (List[int]): list of sequence lengths
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    """

    # loop for each sequence
    all_velocities = []
    all_times = []
    seq_lens = []
    crop = []
    for filename in seq:
        # determine path to gt file
        dir = filename
        if dir.endswith('.txt'): 
            dir = dir[:-4]  # assumes last four characters are '.txt'
        print("dir", dir)
        if dim == 3:
            if aeva:
                print("Using AEVA velocities for 3D evaluation")
                filepath = osp.join(
                    path, dir, "applanix/aeva_poses.csv"
                ) # use 'aeva_poses.csv' for groundtruth
                T_calib = np.loadtxt(osp.join(path, dir, "calib/T_applanix_aeva.txt"))
            else:
                filepath = osp.join(
                    path, dir, "applanix/lidar_poses.csv"
                )  # use 'lidar_poses.csv' for groundtruth
                T_calib = np.loadtxt(osp.join(path, dir, "calib/T_applanix_lidar.txt"))
            velocities, times = read_vel_file_gt(filepath, T_calib, dim)
            times_np = np.stack(times)

            filepath = osp.join(path, dir, 'applanix/camera_poses.csv')  # read in timestamps of camera groundtruth
            _, ctimes = read_vel_file_gt(filepath, np.identity(4), dim)
            if len(ctimes) == 0:
                # Something went wrong with loading camera timestamps, throw error
                raise ValueError(f"No camera timestamps found for sequence {dir}.")
            istart = np.searchsorted(times_np, ctimes[0])
            iend = np.searchsorted(times_np, ctimes[-1])
            velocities = velocities[istart:iend]
            times = times[istart:iend]
            crop += [(istart, iend)]

        elif dim == 2:
            filepath = osp.join(
                path, dir, "applanix/radar_poses.csv"
            )  # use 'radar_poses.csv' for groundtruth
            T_calib = np.identity(4)
            velocities, times = read_vel_file_gt(filepath, T_calib, dim)
            crop += [(0, len(velocities))]
        else:
            raise ValueError(
                "Invalid dim value in get_sequence_poses_gt. Use either 2 or 3."
            )

        seq_lens.append(len(times))
        all_velocities.extend(velocities)
        all_times.extend(times)

    return all_velocities, all_times, seq_lens, crop


def read_vel_file_gt(path, T_ab, dim):
    """Reads velocity from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Velocities read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses (from world to sensor frame)
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as f:
        lines = f.readlines()
    velocities = []
    times = []

    T_ab = enforce_orthog(T_ab)
    for line in lines[1:]:
        vel, time = convert_line_to_vel(line, dim)
        vel[:3] = T_ab[:3, :3] @ vel[:3]
        vel[3:] = T_ab[:3, :3] @ vel[3:]
        # If 2D, zero out z, roll, pitch
        if dim == 2:
            vel[2:5] = 0.0
        velocities += [
            vel
        ]  # convert T_iv to T_vi and apply calibration
        times += [int(time)]  # microseconds
    return velocities, times


def convert_line_to_vel(line, dim=3):
    """Reads velocities from list of strings (single row of the comma-separeted groundtruth file). See Boreas
    documentation for format
    Args:
        line (List[string]): list of strings
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (np.ndarray): 4x4 SE(3) pose
        (int): time in nanoseconds
    """
    # returns T_iv
    line = line.replace("\n", ",").split(",")
    line = [float(i) for i in line[:-1]]

    # Get pose first
    # x, y, z -> 1, 2, 3
    # roll, pitch, yaw -> 7, 8, 9
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = line[1]  # x
    T[1, 3] = line[2]  # y
    if dim == 3:
        T[2, 3] = line[3]  # z
        T[:3, :3] = yawPitchRollToRot(line[9], line[8], line[7])
    elif dim == 2:
        T[:3, :3] = yawPitchRollToRot(
            line[9],
            np.round(line[8] / np.pi) * np.pi,
            np.round(line[7] / np.pi) * np.pi,
        )
    else:
        raise ValueError(
            "Invalid dim value in convert_line_to_pose. Use either 2 or 3."
        )

    vbar = np.array([line[4], line[5], line[6]]).reshape(3, 1)
    vbar = np.matmul(T[:3, :3].T, vbar).squeeze()
    body_rate = np.array(
        [vbar[0], vbar[1], vbar[2], line[12], line[11], line[10]]
    ).reshape(6, 1)

    time = int(line[0])
    return body_rate, time

def compute_vel_metrics(vel_gt, vel_pred, times_pred, seq, pred_vel_path, dim, crop):
    """Evaluates velocity metrics for the given sequences and plots.
    Args:
        vel_gt (List[np.ndarray]): List of 6x1 groundtruth velocities
        vel_pred (List[np.ndarray]): List of 6x1 predicted velocities
        times_pred (List[int]): List of times (microseconds) corresponding to T_pred
        seq (List[string]): List of sequence file names
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
        crop (List[Tuple]): sequences are cropped to prevent extrapolation, this list holds start and end indices
    Returns:
        vel_RMSE: RMSE of velocity error
        vel_mean: Mean velocity error
        vel_RMSE_out: RMSE of velocity error with outliers rejected
        vel_mean_out: Mean velocity error with outliers rejected
    """
    vel_pred = np.array(vel_pred)
    vel_gt = np.array(vel_gt)
    times_pred = np.array(times_pred)

    vel_err = []
    for ii, seq_ii in enumerate(seq):
        print("processing vel for sequence", seq_ii, "...")
        vel_pred_ii = vel_pred[:,:,ii]
        vel_gt_ii = vel_gt[:,:,ii]
        if len(vel_pred_ii) != len(vel_gt_ii):
            vel_pred_ii = vel_pred_ii[crop[ii][0] : crop[ii][1], :]

        # Convert to degrees/s
        vel_pred_ii[:, 3:6] = vel_pred_ii[:, 3:6] * 180 / np.pi
        vel_gt_ii[:, 3:6] = vel_gt_ii[:, 3:6] * 180 / np.pi

        v_err_ii = vel_pred_ii - vel_gt_ii

        if dim == 2:
            v_err_ii[:, 2:5] = 0.0

        vel_err += [v_err_ii]
        times_ii = times_pred[crop[ii][0] : crop[ii][1]] / 1e6
        times_ii = times_ii - times_ii[0]

        plot_vel_stats(seq_ii, pred_vel_path, vel_pred_ii, vel_gt_ii, v_err_ii, times_ii)

    vel_err = np.array(vel_err).reshape(-1, 6)
    vel_RMSE = np.sqrt(np.mean(np.array(vel_err) ** 2, axis=0))
    vel_mean = np.mean(vel_err, axis=0)
    
    # Compute outlier rejected RMSE and mean
    outlier_thres = 2.0
    vel_err_out = vel_err[np.all(np.abs(vel_err[:, :3]) < outlier_thres, axis=1)]
    if vel_err_out.shape[0] < vel_err.shape[0]:
        outlier_timestamps = times_pred[np.where(np.any(np.abs(vel_err[:, :3]) > outlier_thres, axis=1))]
        print("Outliers at scan num, timestamps, error:")
        print("*commented out right now*")
        for ts in outlier_timestamps:
            rad_num = np.where(times_pred == ts)[0][0]
            #print(rad_num, ts, vel_err[np.where(times_pred == ts)][0,:2])
        print("Num outliers rejected:", len(outlier_timestamps))
    vel_RMSE_out = np.sqrt(np.mean(np.array(vel_err_out) ** 2, axis=0))
    vel_mean_out = np.mean(vel_err_out, axis=0)

    if dim == 2:
        # Crop out z, roll, pitch
        vel_RMSE = np.array([vel_RMSE[0], vel_RMSE[1], vel_RMSE[5]])
        vel_mean = np.array([vel_mean[0], vel_mean[1], vel_mean[5]])

        vel_RMSE_out = np.array([vel_RMSE_out[0], vel_RMSE_out[1], vel_RMSE_out[5]])
        vel_mean_out = np.array([vel_mean_out[0], vel_mean_out[1], vel_mean_out[5]])

    return vel_RMSE, vel_mean, vel_RMSE_out, vel_mean_out


def plot_vel_stats(seq, dir, vel_pred, vel_gt, v_err, times_ii):
    """Outputs plots of calculated statistics to specified directory.
    Args:
        seq (List[string]): list of sequence file names
        dir (string): directory path for plot outputs
        vel_pred (List[np.ndarray]): List of 6x1 predicted velocities
        vel_gt (List[np.ndarray]): List of 6x1 groundtruth velocities
        v_err (List[np.ndarray]): List of 6x1 velocity errors
    """
    # Plot superimposed velocities in a 2x3 grid: linear velocities (vx, vy, vz) in first row, angular velocities (wx, wy, wz) in second row
    fig, axs = plt.subplots(2, 3, figsize=(18, 6))

    # Linear velocities
    axs[0, 0].plot(times_ii, vel_gt[:, 0], label="gt")
    axs[0, 0].plot(times_ii, vel_pred[:, 0], label="pred")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Forward Velocity [m/s]")
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].set_title("vx")

    axs[0, 1].plot(times_ii, vel_gt[:, 1], label="gt")
    axs[0, 1].plot(times_ii, vel_pred[:, 1], label="pred")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Side Velocity [m/s]")
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].set_title("vy")

    axs[0, 2].plot(times_ii, vel_gt[:, 2], label="gt")
    axs[0, 2].plot(times_ii, vel_pred[:, 2], label="pred")
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Vertical Velocity [m/s]")
    axs[0, 2].legend(loc='upper right')
    axs[0, 2].set_title("vz")

    # Angular velocities
    axs[1, 0].plot(times_ii, vel_gt[:, 3], label="gt")
    axs[1, 0].plot(times_ii, vel_pred[:, 3], label="pred")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Roll Rate [deg/s]")
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].set_title("wx")

    axs[1, 1].plot(times_ii, vel_gt[:, 4], label="gt")
    axs[1, 1].plot(times_ii, vel_pred[:, 4], label="pred")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Pitch Rate [deg/s]")
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_title("wy")

    axs[1, 2].plot(times_ii, vel_gt[:, 5], label="gt")
    axs[1, 2].plot(times_ii, vel_pred[:, 5], label="pred")
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Yaw Rate [deg/s]")
    axs[1, 2].legend(loc='upper right')
    axs[1, 2].set_title("wz")

    plt.tight_layout()

    plt.savefig(osp.join(dir, seq[:-4] + "_vel.pdf"), pad_inches=0, bbox_inches='tight')
    plt.close()

    # Plot errors
    fig, axs = plt.subplots(1, 3, figsize=(18, 3))
    axs[0].plot(times_ii, v_err[:,0])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Forward Velocity Error [m/s]")

    axs[1].plot(times_ii, v_err[:,1], label="gt")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Side Velocity [m/s]")

    axs[2].plot(times_ii, v_err[:,5], label="gt")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Yaw Velocity [deg/s]")

    plt.savefig(osp.join(dir, seq[:-4] + "_vel_err.pdf"), pad_inches=0, bbox_inches='tight')
    plt.close()

    # Plot error histograms
    fig, axs = plt.subplots(1, 3, figsize=(18, 3))
    axs[0].hist(v_err[:,0], bins=100)
    axs[0].set_xlabel("Error [m/s]")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Fwd. Velocity Error Histogram")

    axs[1].hist(v_err[:,1], bins=100)
    axs[1].set_xlabel("Error [m/s]")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Side Velocity Error Histogram")

    axs[2].hist(v_err[:,5], bins=100)
    axs[2].set_xlabel("Error [deg/s]")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Yaw Velocity Error Histogram")

    plt.savefig(osp.join(dir, seq[:-4] + "_vel_err_hist.pdf"), pad_inches=0, bbox_inches='tight')
    plt.close()

    # Plot error as a function of ground truth yaw
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    axs[0].plot(vel_gt[:,5], v_err[:,0], '.')
    axs[0].set_xlabel("Linear Velocity [m/s]")
    axs[0].set_ylabel("Error")
    axs[0].set_title("Fwd. Velocity Error vs. Ground Truth Angular velocity")

    axs[1].plot(vel_gt[:,5], v_err[:,1], '.')
    axs[1].set_xlabel("Linear Velocity [m/s]")
    axs[1].set_ylabel("Error")
    axs[1].set_title("Side Velocity Error vs. Ground Truth Angular velocity")

    plt.savefig(osp.join(dir, seq[:-4] + "_vel_err_vs_gt_yaw.pdf"), pad_inches=0, bbox_inches='tight')
    plt.close()

    # Plot error as a function of ground truth velocity
    fig, axs = plt.subplots(1, 3, figsize=(18, 3))
    axs[0].plot(vel_gt[:,0], v_err[:,0], '.')
    axs[0].set_xlabel("Linear Velocity [m/s]")
    axs[0].set_ylabel("Error")
    axs[0].set_title("Fwd. Velocity Error vs. Ground Truth Fwd. Velocity")

    axs[1].plot(vel_gt[:,1], v_err[:,1], '.')
    axs[1].set_xlabel("Linear Velocity [m/s]")
    axs[1].set_ylabel("Error")
    axs[1].set_title("Side Velocity Error vs. Ground Truth Side Velocity")

    axs[2].plot(vel_gt[:,5], v_err[:,5], '.')
    axs[2].set_xlabel("Linear Velocity [m/s]")
    axs[2].set_ylabel("Error")
    axs[2].set_title("Yaw Velocity Error vs. Ground Truth Yaw Velocity")

    plt.savefig(osp.join(dir, seq[:-4] + "_vel_err_vs_gt_vel.pdf"), pad_inches=0, bbox_inches='tight')
    plt.close()




    
