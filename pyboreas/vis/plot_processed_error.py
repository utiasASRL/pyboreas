import argparse
import os.path as osp
import struct
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Periods during which daylight savings is in effect in Toronto
dst = [
    (1583650800, 1604210399),
    (1615705200, 1636264799),
    (1647154800, 1667714399),
    (1678604400, 1699163999),
    (1710054000, 1730613599),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/mnt/data1/glen_shields/2020_12_01/",
        type=str,
        help="location of root folder which contains applanix/smrmsg.out",
    )
    args = parser.parse_args()
    with open(osp.join(args.root, "applanix", "smrmsg.out"), "rb") as f:
        fc = f.read()
    with open(osp.join(args.root, "applanix", "ros_and_gps_time.csv"), "r") as f:
        lines = f.readlines()
        start_time = float(lines[1].split(",")[1]) - 5.0
        end_time = float(lines[-1].split(",")[1]) + 5.0
    dt = datetime.fromtimestamp(start_time)
    g2 = (
        dt.isoweekday() * 24 * 3600
        + dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond * 1e-6
    )
    start_week = round(start_time - g2)
    # get timezone offset:
    # Toronto time is GMT-4 or GMT-5 depending on time of year
    time_zone_offset = 5 * 3600
    for period in dst:
        if period[0] < start_time and start_time < period[1]:
            time_zone_offset = 4 * 3600

    start_gps = start_time + time_zone_offset - start_week
    end_gps = end_time + time_zone_offset - start_week

    t = []
    n = []
    e = []
    d = []
    vn = []
    ve = []
    vd = []
    r = []
    p = []
    h = []

    size = 10 * 8  # size of each line in bytes
    for i in range(len(fc) // size):
        data = struct.unpack("d" * 10, fc[i * size : (i + 1) * size])
        if data[0] < start_gps or data[0] > end_gps:
            continue
        t.append(data[0])
        n.append(data[1])
        e.append(data[2])
        d.append(data[3])
        vn.append(data[4])
        ve.append(data[5])
        vd.append(data[6])
        r.append(data[7])
        p.append(data[8])
        h.append(data[9])

    t = np.array(t)
    t -= t[0]

    matplotlib.rcParams.update(
        {
            "font.size": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 2,
            "font.family": "serif",
            "pdf.fonttype": 42,
        }
    )

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(t, n, label="North", color="r", linewidth=3)
    axs[0].plot(t, e, label="East", color="g", linewidth=3)
    axs[0].plot(t, d, label="Down", color="b", linewidth=3)
    axs[0].grid(which="both", linestyle="--", alpha=1.0)
    axs[0].set_ylabel("Pos. Error (m)", fontsize=20)
    axs[0].legend(loc="upper left", prop={"size": 16})

    axs[1].plot(t, vn, label="V North", color="r", linewidth=3)
    axs[1].plot(t, ve, label="V East", color="g", linewidth=3)
    axs[1].plot(t, vd, label="V Down", color="b", linewidth=3)
    axs[1].grid(which="both", linestyle="--", alpha=1.0)
    axs[1].set_ylabel("Vel. Error (m / s)", fontsize=20)
    axs[1].legend(loc="upper left", prop={"size": 16})

    axs[2].plot(t, r, label="Roll", color="r", linewidth=3)
    axs[2].plot(t, p, label="Pitch", color="g", linewidth=3)
    axs[2].plot(t, h, label="Heading", color="b", linewidth=3)
    axs[2].grid(which="both", linestyle="--", alpha=1.0)
    axs[2].set_xlabel("time (seconds)", fontsize=20)
    axs[2].set_ylabel("Ori. Error\n(arc-minutes)", fontsize=20)
    axs[2].legend(loc="upper left", prop={"size": 16})

    plt.savefig(
        osp.join(args.root, "applanix", "gps_error.pdf"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
