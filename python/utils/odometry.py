import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pylgmath as lgmath
import pysteam as steam
from python.utils.utils import get_inverse_tf, rotationError, translationError, enforce_orthog
from itertools import accumulate
from pylgmath import Transformation
from pysteam.trajectory import Time, TrajectoryInterface
from pysteam.state import TransformStateVar, VectorSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.solver import GaussNewtonSolver
from pysteam.evaluator import TransformStateEvaluator

class TrajStateVar:
    """This class defines a trajectory state variable."""
    def __init__(
          self,
          time: Time,
          pose: TransformStateVar,
          velocity: VectorSpaceStateVar,
    ) -> None:
        self.time: Time = time
        self.pose: TransformStateVar = pose
        self.velocity: VectorSpaceStateVar = velocity

def interpolatePoses(poses, times, query_times):
    """Runs a steam optimization with locked poses and outputs poses queried at query_times
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames)
        times (List[int]): list of times for poses (float for seconds or int for nanoseconds)
        query_times (List[float or int]): list of query times (int for nanoseconds)
    Returns:
        (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames) at query_times
    """
    # smoothing factor diagonal
    Qc_inv = np.diag(1 / np.array([1.0, 0.001, 0.001, 0.001, 0.001, 1.0]))

    # steam state variables
    init_velocity = np.zeros((6, 1))
    states = [
        TrajStateVar(
            Time(nsecs=times[i]),
            TransformStateVar(Transformation(T_ba=enforce_orthog(poses[i])), copy=True),
            VectorSpaceStateVar(init_velocity, copy=True),
        ) for i in range(len(poses))
    ]

    # setup trajectory
    traj = TrajectoryInterface(Qc_inv=Qc_inv, allow_extrapolation=True)
    for state in states:
        traj.add_knot(time=state.time, T_k0=TransformStateEvaluator(state.pose), velocity=state.velocity)
        state.pose.set_lock(True)   # lock all pose variables

    # construct the optimization problem
    opt_prob = OptimizationProblem()
    opt_prob.add_cost_term(*traj.get_prior_cost_terms())
    opt_prob.add_state_var(*[j for i in states for j in (i.pose, i.velocity)])

    # construct the solver
    optimizer = GaussNewtonSolver(opt_prob, verbose=False)

    # solve the problem (states are automatically updated)
    optimizer.optimize()

    query_poses = []
    for time in query_times:
        interp_eval = traj.get_interp_pose_eval(Time(nsecs=time))
        query_poses += [interp_eval.evaluate().matrix()]

    return query_poses

def trajectoryDistances(poses):
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
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2 + dz**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
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

def calcSequenceErrors(poses_gt, poses_pred, step_size=4):
    """Calculate the translation and rotation error for each subsequence across several different lengths.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix, ground truth transforms
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix, predicted transforms
    Returns:
        err (List[Tuple]) each entry in list is [first_frame, r_err, t_err, length, speed]
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err, lengths

def getStats(err, lengths):
    """Computes the average translation and rotation within a sequence (across subsequences of diff lengths)."""
    t_err = 0
    r_err = 0
    len2id = {x: i for i, x in enumerate(lengths)}
    t_err_len = [0.0]*len(len2id)
    r_err_len = [0.0]*len(len2id)
    len_count = [0]*len(len2id)
    for e in err:
        t_err += e[2]
        r_err += e[1]
        j = len2id[e[3]]
        t_err_len[j] += e[2]
        r_err_len[j] += e[1]
        len_count[j] += 1
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err * 100, r_err * 180 / np.pi, [a/float(b) * 100 for a, b in zip(t_err_len, len_count)], \
           [a/float(b) * 180 / np.pi for a, b in zip(r_err_len, len_count)]

def plotStats(seq, root, T_odom, T_gt, lengths, t_err, r_err):
    path_odom = getPathFromTviList(T_odom)
    path_gt = getPathFromTviList(T_gt)

    # plot of path
    plt.figure(figsize=(6, 6))
    plt.plot(path_odom[:, 0], path_odom[:, 1], 'b', linewidth=0.5, label='Estimate')
    plt.plot(path_gt[:, 0], path_gt[:, 1], 'r', linewidth=0.5, label='Groundtruth')
    plt.plot(path_gt[0, 0], path_gt[0, 1], 'ks', markerfacecolor='none', label='Sequence Start')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(root, seq[:-4] + '_path.pdf'), bbox_inches='tight')
    plt.close()

    # plot of translation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, t_err, 'bs', markerfacecolor='none')
    plt.plot(lengths, t_err, 'b')
    plt.xlabel('Path Length [m]')
    plt.ylabel('Translation Error [%]')
    plt.savefig(os.path.join(root, seq[:-4] + '_tl.pdf'), bbox_inches='tight')
    plt.close()

    # plot of rotation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, r_err, 'bs', markerfacecolor='none')
    plt.plot(lengths, r_err, 'b')
    plt.xlabel('Path Length [m]')
    plt.ylabel('Rotation Error [deg/m]')
    plt.savefig(os.path.join(root, seq[:-4] + '_rl.pdf'), bbox_inches='tight')
    plt.close()

def getPathFromTviList(Tvi_list):
    path = np.zeros((len(Tvi_list), 3), dtype=np.float32)
    for j, Tvi in enumerate(Tvi_list):
        path[j] = (-Tvi[:3, :3].T @ Tvi[:3, 3:4]).squeeze()
    return path

def computeKittiMetrics(T_gt, T_pred, times_gt, times_pred, seq_lens_gt, seq_lens_pred, seq, root, step_size=10):
    """Computes the translational (%) and rotational drift (deg/m) in the KITTI style.
        KITTI rotation and translation metrics are computed for each sequence individually and then
        averaged across the sequences.
    Args:
        T_gt (List[np.ndarray]): List of 4x4 homogeneous transforms (fixed reference frame to frame t)
        T_pred (List[np.ndarray]): List of 4x4 homogeneous transforms (fixed reference frame to frame t)
        seq_lens (List[int]): List of sequence lengths
    Returns:
        t_err: Average KITTI Translation ERROR (%)
        r_err: Average KITTI Rotation Error (deg / m)
    """
    # get start and end indices of each sequence
    indices_gt = [0]
    indices_gt.extend(list(accumulate(seq_lens_gt)))
    indices_pred = [0]
    indices_pred.extend(list(accumulate(seq_lens_pred)))

    # loop for each sequence
    err_list = []
    for i in range(len(seq_lens_pred)):
        # get poses and times of current sequence
        T_gt_seq = T_gt[indices_gt[i]:indices_gt[i+1]]
        T_pred_seq = T_pred[indices_pred[i]:indices_pred[i+1]]
        times_gt_seq = times_gt[indices_gt[i]:indices_gt[i+1]]
        times_pred_seq = times_pred[indices_pred[i]:indices_pred[i+1]]

        # query predicted trajectory at groundtruth times
        T_query = interpolatePoses(T_pred_seq, times_pred_seq, times_gt_seq)

        err, path_lengths = calcSequenceErrors(T_gt_seq, T_query, step_size)
        t_err, r_err, t_err_len, r_err_len = getStats(err, path_lengths)
        err_list.append([t_err, r_err])
        plotStats(seq[i], root, T_query, T_gt_seq, path_lengths, t_err_len, r_err_len)
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    # return t_err * 100, r_err * 180 / np.pi
    return t_err, r_err

def get_sequences(path, prefix=''):
    """Retrieves a list of all the sequences in the dataset with the given prefix."""
    sequences = [f for f in os.listdir(path) if prefix in f]
    sequences.sort()
    return sequences

def get_sequence_poses(path, seq):
    """Retrieves a list of the poses corresponding to the given sequences in the given file path."""

    # loop for each sequence
    all_poses = []
    all_times = []
    seq_lens = []
    for filename in seq:
        # parse file for list of poses and times
        poses, times = read_traj_file(os.path.join(path, filename))
        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)

    return all_poses, all_times, seq_lens

def write_traj_file(path, poses, times):
    """Writes trajectory into a space-separated txt file
    Args:
        path (string): file path including file name
        poses (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames)
        times (List[int]): list of times for poses (int for nanoseconds)
    """
    with open(path, "w") as file:
        # Writing each time (nanoseconds) and pose to file
        for time, pose in zip(times, poses):
            line = [time]
            line.extend(pose.reshape(16)[:12].tolist())
            file.write(' '.join(str(num) for num in line))
            file.write('\n')


def read_traj_file(path):
    """Writes trajectory into a space-separated txt file
    Args:
        path (string): file path including file name
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in nanoseconds
    """
    with open(path, "r") as file:
        # read each time and pose to lists
        poses = []
        times = []

        for line in file:
            line_split = line.strip().split()
            values = [float(v) for v in line_split[1:]]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(pose)
            times.append(int(line_split[0]))

    return poses, times

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='config/steam.json', type=str, help='path to prediction files')
    parser.add_argument('--gt', default=None, type=str, help='path to groundtruth files')
    args = parser.parse_args()

    # parse sequences
    seq = get_sequences(args.pred, '.txt')
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(args.pred, seq)
    T_gt, times_gt, seq_lens_gt = get_sequence_poses(args.gt, seq)

    # compute errors
    t_err, r_err = computeKittiMetrics(T_gt, T_pred, times_gt, times_pred,
                                       seq_lens_gt, seq_lens_pred, seq, args.pred, step_size=10)

    # print out results
    print('Evaluated sequences: ', seq)
    print('Overall error: ', t_err, ' %, ', r_err, ' deg/m')
