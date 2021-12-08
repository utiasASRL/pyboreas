import os
import os.path as osp
from pathlib import Path
import numpy as np
import argparse
from pyboreas.eval.localization import eval_local
from pyboreas.utils.utils import get_inverse_tf, rotToYawPitchRoll, \
	yawPitchRollToRot, rotation_error, translation_error, se3ToSE3, SE3Tose3
from pyboreas.utils.odometry import read_traj_file_gt2

# gt: 4 x 4 x N
# pred: 4 x 4
def get_matching_pose(pred, gt):
	return np.argmin((gt[0, 3, :] - pred[0, 3])**2 + \
		(gt[1, 3, :] - pred[1, 3])**2, axis=0)

def gen_fake_submission(gtpath, gt_ref_seq, ref, seq, predpath, dim=3):
	gt_ref_poses, gt_ref_times = read_traj_file_gt2(osp.join(gt_ref_seq,
		'applanix', ref + '_poses.csv'), dim=dim)
	pred_poses, pred_times = read_traj_file_gt2(osp.join(gtpath,
		seq, 'applanix', ref + '_poses.csv'), dim=dim)
	np.random.seed(42)
	gt = np.stack(gt_ref_poses, -1)
	f = open(osp.join(predpath, seq + '.txt'), 'w')
	for i, p in enumerate(pred_poses):
		idx = get_matching_pose(p, gt)
		T_s1_s2 = get_inverse_tf(gt_ref_poses[idx]) @ p
		xi = np.zeros((6, 1))
		xi[0, 0] = np.random.normal(0, np.sqrt(0.1))
		xi[1, 0] = np.random.normal(0, np.sqrt(0.1))
		xi[2, 0] = np.random.normal(0, np.sqrt(0.1))
		xi[3, 0] = np.random.normal(0, np.sqrt(0.01))
		xi[4, 0] = np.random.normal(0, np.sqrt(0.01))
		xi[5, 0] = np.random.normal(0, np.sqrt(0.01))
		T_s1_s2 = se3ToSE3(xi) @ T_s1_s2
		s = '{} {} '.format(pred_times[i], gt_ref_times[idx])
		t = [str(x) for x in T_s1_s2.reshape(-1)[:12]]
		s += ' '.join(t)
		s += ' '
		cov = np.identity(6)
		cov[0, 0] = 1 / 0.1
		cov[1, 1] = 1 / 0.1
		cov[2, 2] = 1 / 0.1
		cov[3, 3] = 1 / 0.01
		cov[4, 4] = 1 / 0.01
		cov[5, 5] = 1 / 0.01
		c = [str(x) for x in cov.reshape(-1)]
		s += ' '.join(c)
		s += '\n'
		f.write(s)
	f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pred', default='test/demo/pred/loc')
	parser.add_argument('--gt', default='test/demo/gt')
	parser.add_argument('--ref_seq', default='test/demo/gt/boreas-2021-08-05-13-34')
	parser.add_argument('--ref', default='lidar')

	args = parser.parse_args()
	print(args.pred)
	print(args.gt)
	print(args.ref_seq)
	print(args.ref)
	gtpath = args.gt
	seqs = sorted([b for b in os.listdir(args.gt) if 'boreas-20' in b])
	ss = Path(args.ref_seq).stem
	if ss in seqs:
		seqs.remove(ss)
	dim = 3
	radar = True if dim == 2 else False

	for seq in seqs:
		if not osp.exists(osp.join(args.pred, seq + '.txt')):
			gen_fake_submission(args.gt, args.ref_seq, args.ref, seq, args.pred, dim=dim)

	results = eval_local(args.pred, args.gt, seqs, args.ref_seq, radar=radar, plot_dir=args.pred)
	print(results)
