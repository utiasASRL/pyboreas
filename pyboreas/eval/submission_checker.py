import argparse
import os
import os.path as osp
import shutil
import zipfile
from pathlib import Path

import yaml

from pyboreas.data.splits import loc_test, odom_test, loc_test_rt, odom_test_rt


def check_yaml(yml):
    print("Checking metadata.yaml formatting...")

    keys = [
        "benchmark",
        "methodname",
        "email",
        "2d",
        "author",
        "papertitle",
        "paperurl",
        "venue",
        "year",
        "runtimeseconds",
        "computer",
    ]
    for key in keys:
        try:
            yml[key]
        except KeyError:
            print("missing key: {}".format(key))
            return False
    if (
        yml["benchmark"] != "odometry"
        and yml["benchmark"] != "localization"
        and yml["benchmark"] != "detection"
    ):
        print("benchmark incorrect")
        return False

    if yml["benchmark"] == "localization":
        try:
            sensor = yml["ref_sensor"][0]
            if sensor not in ["camera", "lidar", "radar"]:
                print("incorrect ref_sensor: {}".format(sensor))
                return False
        except KeyError:
            print("missing key: ref_sensor, see localization.md for instructions")
            return False

        try:
            sensor = yml["test_sensor"][0]
            if sensor not in ["camera", "lidar", "radar"]:
                print("incorrect test_sensor: {}".format(sensor))
                return False
        except KeyError:
            print("missing key: test_sensor, see localization.md for instructions")
            return False

    if yml["benchmark"] == "odometry":
        try:
            sensors = yml["sensors"]
            for sensor in sensors:
                if sensor not in ["camera", "lidar", "radar", "IMU"]:
                    print("incorrect list of sensors: {}".format(yml["sensors"]))
                    return False
        except KeyError:
            print("missing key: sensors, see odometry.md for instructions")
            return False

    if not isinstance(yml["2d"], bool):
        print("2d must be bool")
        return False

    if len(yml["methodname"]) > 30 or yml["methodname"] is None:
        print("bad methodname")
        return False

    # check length of metadata
    limits = {
        "author": 100,
        "email": 100,
        "papertitle": 150,
        "paperurl": 500,
        "venue": 10,
        "year": 4,
        "runtimeseconds": 10,
        "computer": 50,
    }
    too_long = []
    for key, max_len in limits.items():
        value = yml.get(key)
        if key in ["year", "runtimeseconds"]:
            value_str = str(value)
        else:
            value_str = "" if value is None else str(value)
        if len(value_str) > max_len:
            too_long.append((key, len(value_str), max_len, value_str))

    if too_long:
        print("metadata too long:")
        for key, actual_len, max_len, value_str in too_long:
            preview = value_str if len(value_str) <= 80 else value_str[:77] + "..."
            print("  - {}: length {} > {} (value: {})".format(key, actual_len, max_len, preview))
        return False
    
    print("\033[92mmetadata.yaml check PASSED\033[0m")
    # make this blue
    print("Ensure the email address in metadata.yaml \033[94m{}\033[0m matches the address used for your account/login.".format(yml["email"]))

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="boreas-test-odometry.zip")
    parser.add_argument("--test_times", default="detection_test_times.txt")
    args = parser.parse_args()

    datasets = ["boreas", "boreasrt"]
    dataset = Path(args.file).stem.split("-")[0]
    if dataset.lower() not in datasets:
        raise Exception("{} not one of: {}. Change filename to format <dataset>-<methodname>-<benchmark>.zip".format(dataset, datasets))

    benchmarks = ["odometry", "localization", "detection"]

    bench = Path(args.file).stem.split("-")[-1].split(".")[0]
    if bench not in benchmarks:
        raise Exception("{} not one of: {}. Change filename to format <dataset>-<methodname>-<benchmark>.zip".format(bench, benchmarks))

    tmp = "tmp_results"
    if not osp.exists(tmp):
        os.mkdir(tmp)
    else:
        shutil.rmtree(tmp)
        os.mkdir(tmp)
    with zipfile.ZipFile(args.file, "r") as zip_ref:
        zip_ref.extractall(tmp)

    files = sorted(os.listdir(tmp))
    if "metadata.yaml" not in files:
        raise Exception("metadata.yaml not found")
    files.remove("metadata.yaml")

    with open(osp.join(tmp, "metadata.yaml")) as f:
        meta = yaml.safe_load(f)

    if not check_yaml(meta):
        raise Exception("metadata.yaml not correctly formatted")
    
    if dataset.lower() != meta["dataset"].lower():
        raise Exception("filename dataset {} != metadata dataset tag {}".format(dataset, meta["dataset"]))

    if bench != meta["benchmark"]:
        raise Exception("filename benchmark {} != metadata benchmark {}".format(bench, meta["benchmark"]))

    if dataset.lower() == "boreas":
        print("Branch: boreas -> {}".format(bench))
        if bench == "odometry":
            gt_seqs = [x[0] for x in odom_test]
            pred_seqs = [f.split(".")[0] for f in files]
            if len(gt_seqs) != len(pred_seqs):
                raise Exception("number of predictions does not match number of gt seqs")
            for pred in pred_seqs:
                if pred not in gt_seqs:
                    raise Exception("pred: {} does not match gt seqs".format(pred))
        elif bench == "localization":
            gt_seqs = [x[0] for x in loc_test]
            pred_seqs = [f.split(".")[0] for f in files]
            if len(gt_seqs) != len(pred_seqs):
                raise Exception("number of predictions does not match number of gt seqs")
            for pred in pred_seqs:
                if pred not in gt_seqs:
                    raise Exception("pred: {} does not match gt seqs".format(pred))
        elif bench == "detection":
            if not osp.exists(args.test_times):
                raise Exception("{} not found".format(args.test_times))
            with open(args.test_times) as f:
                gt_times = sorted([line.strip() for line in f.readlines()])
            preds = sorted([f.split(".")[0] for f in files])
            if len(preds) != len(gt_times):
                raise Exception("number of predictions does not match number of gt times")
            for pred in preds:
                if pred not in gt_times:
                    raise Exception("pred: {} does not match a gt time".format(pred))
                
    if dataset.lower() == "boreasrt":
        print("Branch: boreasrt -> {}".format(bench))
        if bench == "odometry":
            gt_seqs = [x[0] for x in odom_test_rt]
            pred_seqs = [f.split(".")[0] for f in files]
            if len(gt_seqs) != len(pred_seqs):
                raise Exception("number of predictions does not match number of gt seqs")
            for pred in pred_seqs:
                if pred not in gt_seqs:
                    raise Exception("pred: {} does not match gt seqs".format(pred))
        elif bench == "localization":
            gt_seqs = [x[0] for x in loc_test_rt]
            pred_seqs = [f.split(".")[0] for f in files]
            if len(gt_seqs) != len(pred_seqs):
                raise Exception("number of predictions does not match number of gt seqs")
            for pred in pred_seqs:
                if pred not in gt_seqs:
                    raise Exception("pred: {} does not match gt seqs".format(pred))

    shutil.rmtree(tmp)
    print("\033[92mSubmission checker PASSED\033[0m")
