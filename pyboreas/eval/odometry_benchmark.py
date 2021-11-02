import argparse
from pyboreas.utils.odometry import get_sequences, get_sequence_poses, compute_kitti_metrics

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='test/demo/pred', type=str, help='path to prediction files')
    parser.add_argument('--gt', default='test/demo/gt', type=str, help='path to groundtruth files')
    args = parser.parse_args()

    # parse sequences
    seq = get_sequences(args.pred, '.txt')
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(args.pred, seq)
    T_gt, times_gt, seq_lens_gt = get_sequence_poses(args.gt, seq)

    # compute errors
    t_err, r_err = compute_kitti_metrics(T_gt, T_pred, times_gt, times_pred,
                                         seq_lens_gt, seq_lens_pred, seq, args.pred, step_size=10)

    # print out results
    print('Evaluated sequences: ', seq)
    print('Overall error: ', t_err, ' %, ', r_err, ' deg/m')