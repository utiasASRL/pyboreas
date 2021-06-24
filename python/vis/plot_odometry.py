import io
import argparse
import pickle
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
from utils.utils import enforce_orthog, get_inverse_tf, get_T_ba

def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)

def convert_plt_to_tensor():
    return ToTensor()(convert_plt_to_img())

def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, T_icra=None, savePDF=False, fnames=None, flip=True):
    """Creates a top-down plot of the predicted odometry results vs. ground truth."""
	# TODO: this needs to be cleaned up with some options removed, documentation todo.
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                'axes.linewidth': 1.5, 'font.family': 'serif', 'pdf.fonttype': 42})
    T_flip = np.identity(4)
    T_flip[1, 1] = -1
    T_flip[2, 2] = -1
    imgs = []
    for seq_i, indices in enumerate(seq_indices):
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        T_icra_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt = []
        y_gt = []
        x_pred = []
        y_pred = []
        x_icra = []
        y_icra = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])
            if T_icra is not None:
                T_icra_ = np.matmul(T_icra[i], T_icra_)
                enforce_orthog(T_icra_)
                T_icra_temp = get_inverse_tf(T_icra_)
                x_icra.append(T_icra_temp[0, 3])
                y_icra.append(T_icra_temp[1, 3])

        plt.figure(figsize=(10, 10), tight_layout=True)
        plt.grid(color='k', which='both', linestyle='--', alpha=0.75, dashes=(8.5, 8.5))
        plt.axes().set_aspect('equal')        
        plt.plot(x_gt, y_gt, 'k', linewidth=2.5, label='GT')
        if x_icra and y_icra:
            plt.plot(x_icra, y_icra, 'r', linewidth=2.5, label='MC-RANSAC')
        plt.plot(x_pred, y_pred, 'b', linewidth=2.5, label='HERO')
        plt.xlabel('x (m)', fontsize=16)
        plt.ylabel('y (m)', fontsize=16)
        plt.legend(loc="upper left", edgecolor='k', fancybox=False)
        if savePDF and fnames is not None:
            plt.savefig(fnames[seq_i], bbox_inches='tight', pad_inches=0.0)
        if returnTensor:
            imgs.append(convert_plt_to_tensor())
        else:
            imgs.append(convert_plt_to_img())
    return imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./', type=str, help='path to odom.obj files')
    parser.add_argument('--sequence', default='2019-01-10-14-02-34-radar-oxford-10k', type=str, help='sequence to plot')
    args = parser.parse_args()
    [T_gt, T_pred] = pickle.load(open(args.root + 'odom' + args.sequence + '.obj', 'rb'))
    fname = args.root + args.sequence + '.pdf'
    plot_sequences(T_gt, T_pred, [len(T_gt)], returnTensor=False, T_icra=None, savePDF=True, fnames=[fname])
