import os
from pathlib import Path
from time import time
from itertools import accumulate, repeat
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from pyboreas.utils.utils import get_inverse_tf, rotation_error, translation_error, enforce_orthog, yawPitchRollToRot, \
    get_time_from_filename
from pylgmath import Transformation
from pysteam.trajectory import Time, TrajectoryInterface
from pysteam.state import TransformStateVar, VectorSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.solver import GaussNewtonSolver
from pysteam.evaluator import TransformStateEvaluator


class TrajStateVar:
    """This class defines a trajectory state variable for steam."""
    def __init__(
          self,
          time: Time,
          pose: TransformStateVar,
          velocity: VectorSpaceStateVar,
    ) -> None:
        self.time: Time = time
        self.pose: TransformStateVar = pose
        self.velocity: VectorSpaceStateVar = velocity


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
    Qc_inv = np.diag(1 / np.array([0.1, 1.0, 0.1, 0.01, 0.01, 0.1]))

    # steam state variables
    states = []
    for i in range(len(poses)):
        if i == 0:
            dt = (times[1] - times[0])*1e-6  # microseconds to seconds
            dT = Transformation(T_ba=poses[1]) @ Transformation(T_ba=poses[0]).inverse()
        else:
            dt = (times[i] - times[i-1])*1e-6  # microseconds to seconds
            dT = Transformation(T_ba=poses[i]) @ Transformation(T_ba=poses[i-1]).inverse()
        velocity = dT.vec() / dt    # initializing with finite difference
        states += [TrajStateVar(
                        Time(nsecs=int(times[i]*1e3)),  # microseconds to nano
                        TransformStateVar(Transformation(T_ba=poses[i])),
                        VectorSpaceStateVar(velocity))]

    # setup trajectory
    traj = TrajectoryInterface(Qc_inv=Qc_inv, allow_extrapolation=True)
    for state in states:
        traj.add_knot(time=state.time, T_k0=TransformStateEvaluator(state.pose), w_0k_ink=state.velocity)
        state.pose.set_lock(True)   # lock all pose variables

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
    for time in query_times:
        interp_eval = traj.get_interp_pose_eval(Time(nsecs=int(time*1e3)))
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
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2 + dz**2))
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


def calc_sequence_errors(poses_gt, poses_pred, step_size):
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
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.1 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
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


def plot_stats(seq, dir, T_odom, T_gt, lengths, t_err, r_err):
    """Outputs plots of calculated statistics to specified directory.
    Args:
        seq (List[string]): list of sequence file names
        dir (string): directory path for plot outputs
        T_odom (List[np.ndarray]): list of 4x4 estimated poses T_vk_i (vehicle frame at time k and fixed frame i)
        T_gt (List[np.ndarray]): List of 4x4 groundtruth poses T_vk_i (vehicle frame at time k and fixed frame i)
        lengths (List[int]): list of lengths that odometry is evaluated at
        t_err (List[float]): list of average translation error corresponding to lengths
        r_err (List[float]): list of average rotation error corresponding to lengths
    """
    path_odom = get_path_from_Tvi_list(T_odom)
    path_gt = get_path_from_Tvi_list(T_gt)

    # plot of path
    plt.figure(figsize=(6, 6))
    plt.plot(path_odom[:, 0], path_odom[:, 1], 'b', linewidth=0.5, label='Estimate')
    plt.plot(path_gt[:, 0], path_gt[:, 1], '--r', linewidth=0.5, label='Groundtruth')
    plt.plot(path_gt[0, 0], path_gt[0, 1], 'ks', markerfacecolor='none', label='Sequence Start')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(dir, seq[:-4] + '_path.pdf'), bbox_inches='tight')
    plt.close()

    # plot of translation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, t_err, 'bs', markerfacecolor='none')
    plt.plot(lengths, t_err, 'b')
    plt.xlabel('Path Length [m]')
    plt.ylabel('Translation Error [%]')
    plt.savefig(os.path.join(dir, seq[:-4] + '_tl.pdf'), bbox_inches='tight')
    plt.close()

    # plot of rotation error along path length
    plt.figure(figsize=(6, 3))
    plt.plot(lengths, r_err, 'bs', markerfacecolor='none')
    plt.plot(lengths, r_err, 'b')
    plt.xlabel('Path Length [m]')
    plt.ylabel('Rotation Error [deg/m]')
    plt.savefig(os.path.join(dir, seq[:-4] + '_rl.pdf'), bbox_inches='tight')
    plt.close()


def get_path_from_Tvi_list(Tvi_list):
    """Gets 3D path (xyz) from list of poses T_vk_i (transform between vehicle frame at time k and fixed frame i).
    Args:
        Tvi_list (List[np.ndarray]): K length list of 4x4 poses T_vk_i
    Returns:
        path (np.ndarray): K x 3 numpy array of xyz coordinates
    """
    path = np.zeros((len(Tvi_list), 3), dtype=np.float64)
    for j, Tvi in enumerate(Tvi_list):
        path[j] = (-Tvi[:3, :3].T @ Tvi[:3, 3:4]).squeeze()
    return path


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
    T_query = interpolate_poses(T_pred, times_pred, times_gt, solver)   # interpolate
    write_traj_file(out_fname, T_query, times_gt)    # write out
    print(f'interpolated sequence {os.path.basename(out_fname)}, output file: {out_fname}')

    return


def compute_interpolation(T_pred, times_gt, times_pred, seq_lens_gt, seq_lens_pred, seq, out_dir, solver, processes):
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
    T_pred_seq =  (T_pred[indices_pred[i]:indices_pred[i+1]] for i in range(len(seq_lens_pred)))
    times_gt_seq = (times_gt[indices_gt[i]:indices_gt[i+1]] for i in range(len(seq_lens_pred)))
    times_pred_seq = (times_pred[indices_pred[i]:indices_pred[i+1]] for i in range(len(seq_lens_pred)))
    out_fname_seq = (os.path.join(out_dir, seq[i]) for i in range(len(seq_lens_pred)))
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
            print('interpolating sequence', seq[i], '...')
            T_query = interpolate_poses(T_pred_i, times_pred_i, times_gt_i, solver)   # interpolate
            write_traj_file(os.path.join(out_dir, seq[i]), T_query, times_gt_i)    # write out
            print(seq[i], 'took', str(time() - ts), ' seconds')
            print('output file:', os.path.join(out_dir, seq[i]), '\n')
    else:
        # compute interpolation for each sequence in parallel
        with Pool(processes) as p:
            ts = time()  # start time

            print(f'interpolating {len(seq_lens_pred)} sequences in parallel using {processes} workers ...')
            p.starmap(
                compute_interpolation_one_seq, zip(T_pred_seq, times_gt_seq, times_pred_seq, out_fname_seq, solver_seq))
            print(f'interpolation took {time() - ts:.2f} seconds\n')

    return


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
        step_size = 4   # every 4 frames should be 1 second
    else:
        raise ValueError('Invalid dim value in compute_kitti_metrics. Use either 2 or 3.')

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
        T_gt_seq = T_gt[indices_gt[i]:indices_gt[i+1]]
        T_pred_seq = T_pred[indices_pred[i]:indices_pred[i+1]]
        # times_gt_seq = times_gt[indices_gt[i]:indices_gt[i+1]]
        # times_pred_seq = times_pred[indices_pred[i]:indices_pred[i+1]]

        print('processing sequence', seq[i], '...')
        if len(T_pred_seq) != len(T_gt_seq):
            T_pred_seq = T_pred_seq[crop[i][0]:crop[i][1]]

        err, path_lengths = calc_sequence_errors(T_gt_seq, T_pred_seq, step_size)
        t_err, r_err, t_err_len, r_err_len = get_stats(err, path_lengths)
        err_list.append([t_err, r_err])

        print(seq[i], 'took', str(time() - ts), ' seconds')
        print('Error: ', t_err, ' %, ', r_err, ' deg/m \n')

        if plot_dir:
            plot_stats(seq[i], plot_dir, T_pred_seq, T_gt_seq, path_lengths, t_err_len, r_err_len)

    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]

    return t_err, r_err, err_list


def get_sequences(path, file_ext=''):
    """Retrieves a list of all the sequences in the dataset with the given prefix.
    Args:
        path (string): directory path to where the files are
        file_ext (string): string identifier to look for (e.g., '.txt')
    Returns:
        sequences (List[string]): list of sequence file names
    """
    sequences = [f for f in os.listdir(path) if file_ext in f]
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
        poses, times = read_traj_file(os.path.join(path, filename))
        seq_lens.append(len(times))
        all_poses.extend(poses)
        all_times.extend(times)

    return all_poses, all_times, seq_lens

def get_sequence_poses_gt(path, seq, dim):
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
        dir = filename[:-4]     # assumes last four characters are '.txt'
        if dim == 3:
            filepath = os.path.join(path, dir, 'applanix/lidar_poses.csv')  # use 'lidar_poses.csv' for groundtruth
            T_calib = np.loadtxt(os.path.join(path, dir, 'calib/T_applanix_lidar.txt'))
            poses, times = read_traj_file_gt(filepath, T_calib, dim)
            times_np = np.stack(times)

            filepath = os.path.join(path, dir, 'applanix/camera_poses.csv')  # read in timestamps of camera groundtruth
            _, ctimes = read_traj_file_gt(filepath, np.identity(4), dim)
            istart = np.searchsorted(times_np, ctimes[0])
            iend = np.searchsorted(times_np, ctimes[-1])
            poses = poses[istart:iend]
            times = times[istart:iend]
            crop += [(istart, iend)]
            if times[0] < ctimes[0] or times[-1] > ctimes[-1]:
                raise ValueError('Invalid start and end indices for groundtruth.')

        elif dim == 2:
            filepath = os.path.join(path, dir, 'applanix/radar_poses.csv')  # use 'radar_poses.csv' for groundtruth
            T_calib = np.identity(4)
            poses, times = read_traj_file_gt(filepath, T_calib, dim)
            crop += [(0, len(poses))]
        else:
            raise ValueError('Invalid dim value in get_sequence_poses_gt. Use either 2 or 3.')

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
        dir = filename[:-4]     # assumes last four characters are '.txt'
        lfilepath = os.path.join(path, dir, 'applanix/lidar_poses.csv')  # use 'lidar_poses.csv' for groundtruth
        cfilepath = os.path.join(path, dir, 'applanix/camera_poses.csv')  # read in timestamps of camera groundtruth
        if os.path.isfile(lfilepath) and os.path.isfile(cfilepath):
            # csv files exist, use them
            _, times = read_traj_file_gt(lfilepath, np.identity(4), dim=3)
            times_np = np.stack(times)
            _, ctimes = read_traj_file_gt(cfilepath, np.identity(4), dim=3)
        else:
            # read timestamps from data
            lpath = os.path.join(path, dir, 'lidar')  # read lidar data filenames
            times = [int(Path(f).stem) for f in os.listdir(lpath) if '.bin' in f]
            times.sort()
            times_np = np.stack(times)

            cpath = os.path.join(path, dir, 'camera')  # read camera data filenames
            ctimes = [int(Path(f).stem) for f in os.listdir(cpath) if '.png' in f]
            ctimes.sort()

        istart = np.searchsorted(times_np, ctimes[0])
        iend = np.searchsorted(times_np, ctimes[-1])
        times = times[istart:iend]
        crop += [(istart, iend)]
        if times[0] < ctimes[0] or times[-1] > ctimes[-1]:
            raise ValueError('Invalid start and end indices for groundtruth.')

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
        for time, pose in zip(times, poses):
            line = [time]
            line.extend(pose.reshape(16)[:12].tolist())
            file.write(' '.join(str(num) for num in line))
            file.write('\n')


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


def read_traj_file_gt(path, T_ab, dim):
    """Reads trajectory from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    poses = []
    times = []

    T_ab = enforce_orthog(T_ab)
    for line in lines[1:]:
        pose, time = convert_line_to_pose(line, dim)
        poses += [enforce_orthog(T_ab @ get_inverse_tf(pose))]  # convert T_iv to T_vi and apply calibration
        times += [int(time)]  # microseconds
    return poses, times

def convert_line_to_pose(line, dim):
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
    line = line.replace('\n', ',').split(',')
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
        T[:3, :3] = yawPitchRollToRot(line[9], 0, 0)
    else:
        raise ValueError('Invalid dim value in convert_line_to_pose. Use either 2 or 3.')
    time = int(line[0])
    return T, time
