import argparse
import os
from pyboreas.utils.odometry import get_sequences, get_sequence_poses, get_sequence_poses_gt, \
    compute_kitti_metrics, compute_interpolation, get_sequence_times_gt

def eval_odom(pred='test/demo/pred/3d',
              gt='test/demo/gt',
              radar=False,
              interp='',
              processes=os.cpu_count(),
              solver=True):

    # evaluation mode
    dim = 2 if radar else 3
    if dim == 2:
        interp = ''  # force interpolation to be off for radar (2D) evaluation

    # parse sequences
    seq = get_sequences(pred, '.txt')
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    if interp:     # if we are interpolating...
        # can't be the same as pred
        if interp == pred:
            raise ValueError('`interp` directory path cannot be the same as the `pred` directory path')

        # get corresponding groundtruth times
        times_gt, seq_lens_gt, _ = get_sequence_times_gt(gt, seq)

        # make interp directory if it doesn't exist
        if not os.path.exists(interp):
            os.mkdir(interp)

        # interpolate
        compute_interpolation(T_pred, times_gt, times_pred, seq_lens_gt, seq_lens_pred, seq, interp, solver, processes)
    else:
        # get corresponding groundtruth poses
        T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

        # compute errors
        t_err, r_err, _ = compute_kitti_metrics(T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop)

        # print out results
        print('Evaluated sequences: ', seq)
        print('Overall error: ', t_err, ' %, ', r_err, ' deg/m')
        return t_err, r_err


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='test/demo/pred/3d', type=str, help='path to prediction files')
    parser.add_argument('--gt', default='test/demo/gt', type=str, help='path to groundtruth files')
    parser.add_argument('--radar', dest='radar', action='store_true', help='evaluate radar odometry in SE(2)')
    parser.set_defaults(radar=False)
    parser.add_argument('--interp', default='', type=str, help='path to interpolation output, do not set if evaluating')
    parser.add_argument('--processes', default=os.cpu_count(), type=int,
                        help='number of workers to use for built-in interpolation')
    parser.add_argument('--no-solver', dest='solver', action='store_false',
                        help='disable solver for built-in interpolation')
    parser.set_defaults(solver=True)
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.radar, args.interp, args.processes, args.solver)