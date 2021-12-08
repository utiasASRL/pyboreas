import os
import os.path as osp
from pathlib import Path
import argparse
import numpy as np
from pyboreas.data.splits import loc_test, loc_reference
from pyboreas.utils.utils import get_inverse_tf, get_closest_index, \
	rotation_error, SE3Tose3
from pyboreas.utils.odometry import read_traj_file2, read_traj_file_gt2, plot_loc_stats

def get_Tas(gtpath, seq, sensor='lidar'):
	T_applanix_lidar = np.loadtxt(osp.join(gtpath, seq, 'calib', 'T_applanix_lidar.txt'))
	if sensor == 'camera':
		T_camera_lidar = np.loadtxt(osp.join(gtpath, seq, 'calib', 'T_camera_lidar.txt'))
		return np.matmul(T_applanix_lidar, get_inverse_tf(T_camera_lidar))
	elif sensor == 'radar':
		T_radar_lidar = np.loadtxt(osp.join(gtpath, seq, 'calib', 'T_radar_lidar.txt'))
		return np.matmul(T_applanix_lidar, get_inverse_tf(T_radar_lidar))
	return T_applanix_lidar

def check_time_match(pred_times, gt_times):
	assert(len(pred_times) == len(gt_times))
	p = np.array(pred_times)
	g = np.array(gt_times)
	assert(np.sum(p - g) == 0)

def check_ref_time_match(ref_times, gt_ref_times):
	indices = np.searchsorted(gt_ref_times, ref_times)
	p = np.array(ref_times)
	g = np.array(gt_ref_times)
	assert(np.sum(g[indices] - p) == 0)

def get_T_enu_s1(query_time, gt_times, gt_poses):
    closest = get_closest_index(query_time, gt_times)
    assert(query_time == gt_times[closest]), 'query: {}'.format(query_time)
    return gt_poses[closest]

def compute_errors(Te):
	return [Te[0, 3], Te[1, 3], Te[2, 3], rotation_error(Te) * 180 / np.pi]

def root_mean_square(errs):
	return np.sqrt(np.mean(np.power(np.array(errs), 2), axis=0)).squeeze()

def eval_local(predpath, gtpath, gt_seqs, gt_ref_seq, radar=False, ref='lidar', plot_dir=None):
	dim = 2 if radar else 3
	pred_files = sorted([f for f in os.listdir(predpath) if '.txt' in f])
	assert(len(pred_files) == len(gt_seqs)), '{} {}'.format(pred_files, gt_seqs)
	for predfile in pred_files:
		if Path(predfile).stem.split('.')[0] not in gt_seqs:
			raise Exception("prediction file doesn't match ground truth sequence list")
	
	gt_ref_poses, gt_ref_times = read_traj_file_gt2(osp.join(gtpath, gt_ref_seq, 'applanix', ref + '_poses.csv'), dim=dim)
	seq_rmse = []
	seq_consist = []
	seqs_have_cov = True
	for predfile, seq in zip(pred_files, gt_seqs):
		print('Processing {}...'.format(seq))
		T_as = get_Tas(gtpath, seq, ref)
		T_sa = get_inverse_tf(T_as)
		pred_poses, pred_times, ref_times, cov_matrices, has_cov = read_traj_file2(osp.join(predpath, predfile))
		seqs_have_cov *= has_cov
		gt_poses, gt_times = read_traj_file_gt2(osp.join(gtpath, seq, 'applanix', ref + '_poses.csv'), dim=dim)
		# check that pred_times is a 1-to-1 match with gt_times
		check_time_match(pred_times, gt_times)
		# check that each ref time matches to one gps_ref_time
		check_ref_time_match(ref_times, gt_ref_times)
		errs = []
		consist = []
		T_gt_seq = []
		T_pred_seq = []
		Xi = []
		Cov = []
		for j, pred_T_s1_s2 in enumerate(pred_poses):
			gt_T_enu_s2 = gt_poses[j]
			T_gt_seq.append(get_inverse_tf(gt_T_enu_s2))

			gt_T_enu_s1 = get_T_enu_s1(ref_times[j], gt_ref_times, gt_ref_poses)
			T_pred_seq.append(get_inverse_tf(gt_T_enu_s1 @ pred_T_s1_s2))

			gt_T_s1_s2 = get_inverse_tf(gt_T_enu_s1) @ gt_T_enu_s2
			T = pred_T_s1_s2 @ get_inverse_tf(gt_T_s1_s2)
			Te = T_as @ T @ T_sa
			errs.append(compute_errors(Te))

			# If the user submitted a covariance matrix, calculate consistency
			if has_cov:
				if abs(np.sum(cov_matrices[j] - np.identity(6))) < 1e-3:
					consist.append(1)
				xi = SE3Tose3(T)
				Xi.append(xi.squeeze())
				c = xi.T @ cov_matrices[j] @ xi
				consist.append(c[0, 0])  # assumes user has uploaded inverse covariance matrices
				Cov.append(1 / cov_matrices[j].diagonal())
		Xi = np.array(Xi)
		print(np.array(consist).shape)
		Cov = np.array(Cov)
		if plot_dir is not None:
			plot_loc_stats(seq, plot_dir, T_pred_seq, T_gt_seq, errs, consist, Xi, Cov, has_cov)
		rmse = root_mean_square(errs)
		seq_rmse.append(rmse)
		print('RMSE: x: {} m y: {} m z: {} m phi: {} deg'.format(rmse[0], rmse[1], rmse[2], rmse[3]))
		c = -1
		if has_cov:
			c = np.sqrt(max(0, np.mean(consist) / 6.0))
			# c = np.mean(np.sqrt(np.array(consist) / 6.0))
			print('Consistency: {}'.format(c))
		seq_consist.append(c)
		print('\n')

	seq_rmse = np.array(seq_rmse)
	rmse = np.mean(seq_rmse, axis=0).squeeze()
	print('Overall RMSE: x: {} m y: {} m z: {} m phi: {} deg'.format(rmse[0], rmse[1], rmse[2], rmse[3]))
	c = -1
	if seqs_have_cov:
		c = np.mean(seq_consist)
		print('Overall Consistency: {}'.format(c))
	return seq_rmse, seq_consist, seqs_have_cov


if __name__ ==  '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pred', type=str, help='path to prediction files')	
	parser.add_argument('--gt', type=str, help='path to groundtruth sequence folders')
	parser.add_argument('--radar', dest='radar', action='store_true', help='evaluate radar odometry in SE(2)')
	parser.add_argument('--ref', default='lidar', type=str, help='Which sensor to use as a reference (camera|lidar|radar)')
	parser.set_defaults(radar=False)
	args = parser.parse_args()
	assert(args.ref in ['camera', 'lidar', 'radar'])
	gt_seqs = [x[0] for x in loc_test]
	gt_ref_seq = loc_reference
	eval_local(args.pred, args.gt, gt_seqs, gt_ref_seq, args.radar, args.ref)
