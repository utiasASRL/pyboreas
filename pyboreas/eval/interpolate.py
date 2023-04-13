import argparse
import os

from pyboreas.utils.odometry import (
    compute_interpolation,
    get_sequence_poses,
    get_sequence_times_gt,
    get_sequences,
)

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
        "--interp", default="", type=str, help="path to interpolation output"
    )
    parser.add_argument(
        "--processes",
        default=os.cpu_count(),
        type=int,
        help="number of workers to use for built-in interpolation",
    )
    parser.add_argument(
        "--no-solver",
        dest="solver",
        action="store_false",
        help="disable solver for built-in interpolation",
    )
    parser.set_defaults(solver=True)
    args = parser.parse_args()

    # parse sequences
    seq = get_sequences(args.pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(args.pred, seq)

    # can't be the same as pred
    if args.interp == args.pred:
        raise ValueError(
            "`interp` directory path cannot be the same as the `pred` directory path"
        )

    # get corresponding groundtruth times
    times_gt, seq_lens_gt, _ = get_sequence_times_gt(args.gt, seq)

    # make interp directory if it doesn't exist
    if not os.path.exists(args.interp):
        os.mkdir(args.interp)

    # interpolate
    compute_interpolation(
        T_pred,
        times_gt,
        times_pred,
        seq_lens_gt,
        seq_lens_pred,
        seq,
        args.interp,
        args.solver,
        args.processes,
    )
