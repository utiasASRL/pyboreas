import argparse
import multiprocessing
import os.path as osp
from multiprocessing import Pool
from time import time

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train

GROUND = 1
BOX3D = 2
CAR = 0
PEDESTRIAN = 1
CYCLIST = 2
MIN_OVERLAP = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
N_SAMPLE_PTS = 41


class PrData:
    def __init__(self, tp=0, fp=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.v = []


def intersection_area(d, g):
    # top-down intersection area
    pd = Polygon(d.pc.points[:4, :2])
    pg = Polygon(g.pc.points[:4, :2])
    return pg.intersection(pd).area


def boxOverlap(d, g, dim=3):
    inter = intersection_area(d, g)
    if dim == 3:
        ymax = min(d.pos[2, 0] + d.extent[2, 0] / 2, g.pos[2, 0] + g.extent[2, 0] / 2)
        ymin = max(d.pos[2, 0] - d.extent[2, 0] / 2, g.pos[2, 0] - g.extent[2, 0] / 2)
        inter *= max(0, ymax - ymin)
    det = d.extent[0, 0] * d.extent[1, 0]
    gt = g.extent[0, 0] * g.extent[1, 0]
    if dim == 3:
        det *= d.extent[2, 0]
        gt *= g.extent[2, 0]
    union = det + gt - inter + 1e-14
    return inter / union


def getThresholds(v, n_gt):
    t = []
    v = sorted(v, reverse=True)
    current_recall = 0
    for i in range(len(v)):
        l_recall = (i + 1) / float(n_gt)
        if i < len(v) - 1:
            r_recall = (i + 2) / float(n_gt)
        else:
            r_recall = l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < (
            len(v) - 1
        ):
            continue
        t.append(v[i])
        current_recall += 1.0 / float(N_SAMPLE_PTS - 1.0)
    return t


# gtbbs, detbbs: list of BoundingBox objects
def computeStatistics(
    current_class, gtbbs, detbbs, metric, minoverlap, thresh=0, overlap=None
):
    assert overlap is not None
    dim = 2 if metric < 2 else 3
    stat = PrData()
    N = len(gtbbs)
    M = len(detbbs)
    assigned_detection = np.zeros(M)
    NO_DETECTION = -np.inf

    n_gt = 0
    for i in range(N):
        if gtbbs[i].label != current_class:
            continue
        n_gt += 1

        det_idx = None
        max_thresh = NO_DETECTION
        max_overlap = 0
        for j in range(M):
            if (
                detbbs[j].label != current_class
                or assigned_detection[j]
                or detbbs[j].score < thresh
            ):
                continue
            if overlap[i, j] == -1:
                overlap[i, j] = boxOverlap(detbbs[j], gtbbs[i], dim)
            if (
                overlap[i, j] > minoverlap[current_class]
                and detbbs[j].score > max_thresh
            ):
                det_idx = j
                max_thresh = detbbs[j].score
            elif (
                overlap[i, j] > minoverlap[current_class]
                and overlap[i, j] > max_overlap
            ):
                det_idx = j
                max_overlap = overlap[i, j]
                max_thresh = 1

        # compute TP, FP, FN
        if max_thresh == NO_DETECTION:
            stat.fn += 1
        else:
            stat.tp += 1
            stat.v.append(detbbs[det_idx].score)
            assigned_detection[det_idx] = True

    for j in range(M):
        if (
            not assigned_detection[j]
            and detbbs[j].label == current_class
            and detbbs[j].score > thresh
        ):
            stat.fp += 1
    return stat, n_gt


def eval_class(
    current_class, groundtruth, detections, metric, minoverlap, savePath=None
):
    # does not support compute_aos
    # Return: precision [],
    overlaps = []  # cache boxOverlap computations
    n_gt = 0
    v = []
    # For all test images do
    N = len(groundtruth)
    for i in range(N):
        # Only evaluate objects of current class
        overlap = -1 * np.ones((len(groundtruth[i].bbs), len(detections[i].bbs)))
        pr_tmp, n = computeStatistics(
            current_class,
            groundtruth[i].bbs,
            detections[i].bbs,
            metric,
            minoverlap,
            overlap=overlap,
        )
        v.extend(pr_tmp.v)
        n_gt += n
        overlaps.append(overlap)
    # Get scores that must be evaluated for recall discretization
    thresholds = getThresholds(v, n_gt)
    T = len(thresholds)

    # Compute TP,FP,FN for relevant scores
    pr = []
    for t in range(T):
        pr.append(PrData())
    for i in range(N):
        for t in range(T):
            tmp, _ = computeStatistics(
                current_class,
                groundtruth[i].bbs,
                detections[i].bbs,
                metric,
                minoverlap,
                thresholds[t],
                overlaps[i],
            )
            pr[t].tp += tmp.tp
            pr[t].fp += tmp.fp
            pr[t].fn += tmp.fn

    # compute recall, precision
    recall = np.zeros(N_SAMPLE_PTS)
    precision = np.zeros(N_SAMPLE_PTS)
    for i in range(T):
        recall[i] = pr[i].tp / float(pr[i].tp + pr[i].fn)
        precision[i] = pr[i].tp / float(pr[i].tp + pr[i].fp)

    for i in range(T):
        precision[i] = np.max(precision[i:])

    if savePath is not None:
        stats = " ".join([str(p) for p in precision])
        with open(savePath, "w") as f:
            f.write(stats + "\n")
    return precision, get_mAP(precision)


def get_mAP(prec):
    return np.mean(prec) * 100


def inject_noise(det):
    for bb in det.bbs:
        sigma = 0.10
        if bb.label == "Pedestrian":
            sigma = 0.04
        if bb.label == "Cyclist":
            sigma = 0.04
        p = np.random.normal(0, sigma, 3).reshape(3, 1)
        dx = abs(p[0, 0])
        dy = abs(p[1, 0])
        dz = abs(p[2, 0])
        alpha = 0.308
        score = 0.0292 / ((dx + alpha) * (dy + alpha) * (dz + alpha))
        bb.pos = bb.pos + p
        bb.pc.points += p.reshape(1, 3)
        bb.score = score


def eval_obj(groundtruth, detections, radar=False, resultsDir=None):
    cpath = None
    ppath = None
    cypath = None
    if resultsDir is not None:
        cpath = osp.join(resultsDir, "car.txt")
        ppath = osp.join(resultsDir, "ped.txt")
        cypath = osp.join(resultsDir, "cyc.txt")
    p1, carmap = eval_class(
        CLASS_NAMES[CAR], groundtruth, detections, BOX3D, MIN_OVERLAP, cpath
    )
    print(CLASS_NAMES[CAR] + " mAP: {} %".format(carmap))
    if resultsDir is not None:
        plot_pr(resultsDir, p1, "Car")
    p2, pedmap = eval_class(
        CLASS_NAMES[PEDESTRIAN], groundtruth, detections, BOX3D, MIN_OVERLAP, ppath
    )
    print(CLASS_NAMES[PEDESTRIAN] + " mAP: {} %".format(pedmap))
    if resultsDir is not None:
        plot_pr(resultsDir, p2, "Pedestrian")
    p3, cycmap = eval_class(
        CLASS_NAMES[CYCLIST], groundtruth, detections, BOX3D, MIN_OVERLAP, cypath
    )
    print(CLASS_NAMES[CYCLIST] + " mAP: {} %".format(cycmap))
    if resultsDir is not None:
        plot_pr(resultsDir, p3, "Cyclist")
    return [carmap, pedmap, cycmap], [p1, p2, p3]


def plot_pr(plot_dir, precision, label=""):
    delta = 1 / float(N_SAMPLE_PTS - 1)
    x = np.arange(0, 1 + delta, delta)
    plt.figure(figsize=(6, 6))
    plt.plot(x, precision, "b", linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis("equal")
    if label != "":
        plt.title(label)
    plt.savefig(osp.join(plot_dir, label + "_detection.pdf"))
    plt.close()


def get_bbs(root, split, labelFolder, noise=False, N=-1):
    bd = BoreasDataset(root, split=split, labelFolder=labelFolder)
    frames = []
    for seq in bd.sequences:
        seq.filter_frames_gt()
        frames.extend(seq.lidar_frames)
    frames.sort(key=lambda x: x.timestamp)
    if N >= 0:
        frames = frames[:N]

    bb_frames = []

    global _load_bb

    def _load_bb(lid):
        if not lid.has_bbs():
            return None
        bbs = lid.get_bounding_boxes()
        if noise:
            inject_noise(bbs)
        return bbs

    pool = Pool(multiprocessing.cpu_count())
    bb_frames = list(pool.map(_load_bb, frames))
    bb_frames = [x for x in bb_frames if x is not None]
    bb_frames.sort(key=lambda x: x.timestamp)
    return bb_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note: ground truth are stored in "labels_detection" under the ground truth sequence
    parser.add_argument(
        "--gt", type=str, help="path to groundtruth sequence", default="test/demo/gt/"
    )
    # Note: predictions are to be stored in a folder under the ground truth sequence, example: labels_pred
    parser.add_argument(
        "--pred", type=str, help="prediction folder name", default="labels_detection"
    )
    parser.add_argument(
        "--radar", dest="radar", action="store_true", help="evaluate BEV detections"
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        action="store_false",
        help="If set, do not inject noise into preds",
    )
    parser.add_argument(
        "--N", type=int, default=100, help="Set to -1 to evaluate all detections"
    )
    parser.set_defaults(radar=False)
    parser.set_defaults(noise=True)
    args = parser.parse_args()

    # We artificially restrict the number of inputs to N to make the demo run faster.
    groundtruth = get_bbs(args.gt, obj_train, "labels_detection", N=args.N)
    # We inject noise purely to simulate detections as a demo, set this to false during evaluations
    detections = get_bbs(args.gt, obj_train, args.pred, noise=args.noise, N=args.N)

    t0 = time()
    eval_obj(groundtruth, detections, radar=args.radar, resultsDir="test/demo/pred/")
    print("Evaluation time: {}".format(time() - t0))
