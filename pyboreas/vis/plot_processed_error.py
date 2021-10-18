import os.path as osp
import argparse
import struct
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Periods during which daylight savings is in effect in Toronto
dst = [(1583632800, 1604196000), (1615687200, 1636250400), (1647136800, 1667700000), (1678586400, 1699236000)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data1/glen_shields/2020_12_01/', type=str,
                        help='location of root folder which contains applanix/smrmsg.out')
    args = parser.parse_args()
    with open(osp.join(args.root, 'applanix', 'smrmsg.out'), 'rb') as f:
        fc = f.read()
    with open(osp.join(args.root, 'applanix', 'ros_and_gps_time.csv'), 'r') as f:
        lines = f.readlines()
        start_time = float(lines[1].split(',')[1]) - 5.0
        end_time = float(lines[-1].split(',')[1]) + 5.0
    dt = datetime.fromtimestamp(start_time)
    g2 = dt.isoweekday() * 24 * 3600 + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6
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
        data = struct.unpack("d" * 10, fc[i * size: (i + 1) * size])
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

    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                'axes.linewidth': 1.5, 'font.family': 'serif', 'pdf.fonttype': 42})

    plt.figure(figsize=(10, 5))
    plt.plot(t, n, label='North position RMS error (m)')
    plt.plot(t, e, label='East position RMS error (m)')
    plt.plot(t, d, label='Down position RMS error (m)')
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.xlabel('time (seconds)', fontsize=16)
    plt.ylabel('Error (m)', fontsize=16)
    plt.legend(loc="upper left", prop={'size': 12})
    plt.xticks(rotation=45, ha="right")
    plt.savefig(osp.join(args.root, 'applanix', 'gps_position_error.pdf'), bbox_inches='tight', pad_inches=0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(t, vn, label='North velocity RMS error (m/s)')
    plt.plot(t, ve, label='East velocity RMS error (m/s)')
    plt.plot(t, vd, label='Down velocity RMS error (m/s)')
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.xlabel('time (seconds)', fontsize=16)
    plt.ylabel('Error (m / s)', fontsize=16)
    plt.legend(loc="upper left", prop={'size': 12})
    plt.xticks(rotation=45, ha="right")
    plt.savefig(osp.join(args.root, 'applanix', 'gps_velocity_error.pdf'), bbox_inches='tight', pad_inches=0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(t, r, label='Roll RMS error (arc-min)')
    plt.plot(t, p, label='Pitch RMS error (arc-min)')
    plt.plot(t, h, label='Heading RMS error (arc-min)')
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.xlabel('time (seconds)', fontsize=16)
    plt.ylabel('Error (arc-minutes)', fontsize=16)
    plt.legend(loc="upper left", prop={'size': 12})
    plt.xticks(rotation=45, ha="right")
    plt.savefig(osp.join(args.root, 'applanix', 'gps_orientation_error.pdf'), bbox_inches='tight', pad_inches=0.0)
