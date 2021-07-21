import argparse
import numpy as np
import os
import pylgmath as lgmath
import pysteam as steam
from python.utils.utils import get_inverse_tf, rotationError, translationError, enforce_orthog

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
        times (List[float or int]): list of times for poses (float for seconds or int for nanoseconds)
        query_times (List[float or int]): list of query times (float for seconds or int for nanoseconds)
    Returns:
        (List[np.ndarray]): list of 4x4 poses (T_v_i vehicle and inertial frames) at query_times
    """
    # smoothing factor diagonal
    Qc_inv = np.diag(1 / np.array([1.0, 0.001, 0.001, 0.001, 0.001, 1.0]))

    # steam state variables
    init_velocity = np.zeros((6, 1))
    states = [
        TrajStateVar(
            Time(times[i]),
            TransformStateVar(Transformation(T_ba=poses[i]), copy=True),
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
        interp_eval = traj.get_interp_pose_eval(Time(time))
        query_poses += [interp_eval.evaluate().matrix()]

    return query_poses

def trajectoryDistances(poses):
    """Calculates path length along the trajectory.
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_2_1 from current to next)
    Returns:
        List[float]: distance along the trajectory, increasing as a function of time / list index
    """
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
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
    return err

def getStats(err):
    """Computes the average translation and rotation within a sequence (across subsequences of diff lengths)."""
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

def computeKittiMetrics(T_gt, T_pred, times_gt, times_pred, seq_lens, step_size=10):
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
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err_list = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        # query pred at gt times
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err = calcSequenceErrors(poses_gt, poses_pred, step_size)
        t_err, r_err = getStats(err)
        err_list.append([t_err, r_err])
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    return t_err * 100, r_err * 180 / np.pi

def get_sequences(path, prefix=''):
    """Retrieves a list of all the sequences in the dataset with the given prefix."""
    sequences = [f for f in os.listdir(path) if prefix in f]
    sequences.sort()
    return sequences

def get_sequence_poses(path, seq):
    """Retrieves a list of the poses corresponding to the given sequences in the given file path."""

    # loop for each sequence
    T = []
    seq_lens = []
    for filename in seq:
        # parse file for list of poses
        file = open(os.path.join(path, filename))
        counter = 0
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            T.append(get_inverse_tf(pose))
            counter += 1
        seq_lens.append(counter)

    return T, seq_lens


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='config/steam.json', type=str, help='path to prediction files')
    parser.add_argument('--gt', default=None, type=str, help='path to groundtruth files')
    args = parser.parse_args()

    # parse sequences
    seq = get_sequences(args.pred)
    T_pred, seq_lens = get_sequence_poses(args.pred, seq[0:1])
    T_gt, _ = get_sequence_poses(args.gt, seq)

    # compute errors
    t_err, r_err = computeKittiMetrics(T_gt, T_pred, seq_lens, 10)

    a = 1

    # print out results
    # TODO
