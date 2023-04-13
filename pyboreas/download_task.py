import argparse
import os
import os.path as osp

from pyboreas.data.splits import loc_test, loc_train, odom_test, odom_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/path/to/data/boreas/",
        help="path to where Boreas sequences will be stored",
    )
    parser.add_argument(
        "--task", default="odometry", help="[odometry|localization|detection]"
    )
    parser.add_argument("--nocamera", action="store_true", default=False)
    parser.add_argument("--noradar", action="store_true", default=False)
    parser.add_argument("--nolidar", action="store_true", default=False)
    parser.add_argument("--trainonly", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    assert args.task in ["odometry", "localization", "detection"]

    split = []
    if args.task == "odometry":
        split += [x[0] for x in odom_train]
        if not args.trainonly:
            split += [x[0] for x in odom_test]
    elif args.task == "localization":
        split += [x[0] for x in loc_train]
        if not args.trainonly:
            split += [x[0] for x in loc_test]
    elif args.task == "detection":
        split += ["boreas-objects-v1"]

    print("Sequences to download:")
    print(split)

    for seq in split:
        seq_root = osp.join(args.root, seq)
        command = (
            "aws s3 sync s3://boreas/"
            + seq
            + " "
            + seq_root
            + " --exclude '*' --include 'applanix/*' --include video.mp4 --include route.html "
        )
        if not args.nocamera:
            command += "--include camera/* "
        if not args.nolidar:
            command += "--include lidar/* "
        if not args.noradar:
            command += "--include radar/* "
        print(command)
        os.system(command)
