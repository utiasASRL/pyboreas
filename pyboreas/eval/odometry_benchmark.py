import argparse
from pyboreas.utils.odometry import get_sequences, get_sequence_poses, get_sequence_poses_gt, compute_kitti_metrics

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='test/demo/pred/3d', type=str, help='path to prediction files')
    parser.add_argument('--gt', default='test/demo/gt', type=str, help='path to groundtruth files')
    parser.add_argument('--radar', dest='radar', action='store_true', help='evaluate radar odometry in SE(2)')
    parser.set_defaults(radar=False)
    parser.add_argument('--interp', dest='interp', action='store_true', help='use built-in interpolation')
    parser.set_defaults(interp=True)
    args = parser.parse_args()

    # evaluation mode
    dim = 2 if args.radar else 3

    # parse sequences
    seq = get_sequences(args.pred, '.txt')
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(args.pred, seq)
    T_gt, times_gt, seq_lens_gt = get_sequence_poses_gt(args.gt, seq, dim)

    if len(T_pred) != len(T_gt) and not args.interp:
        raise NotImplementedError('Length of predicted trajectories must be the same as groundtruth, '
                                  'unless the built-in interpolation feature is on (--interp True).')

    # compute errors
    t_err, r_err = compute_kitti_metrics(T_gt, T_pred, times_gt, times_pred,
                                         seq_lens_gt, seq_lens_pred, seq, args.pred, dim, args.interp)

    # print out results
    print('Evaluated sequences: ', seq)
    print('Overall error: ', t_err, ' %, ', r_err, ' deg/m')