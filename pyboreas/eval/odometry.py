import argparse

from pyboreas.utils.odometry import (
    compute_kitti_metrics,
    get_sequence_poses,
    get_sequence_poses_gt,
    get_sequences,
)


def eval_odom(pred="test/demo/pred/3d", gt="test/demo/gt", radar=False):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop
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
        "--radar",
        dest="radar",
        action="store_true",
        help="evaluate radar odometry in SE(2)",
    )
    parser.set_defaults(radar=False)
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.radar)
