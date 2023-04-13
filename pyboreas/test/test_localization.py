import os.path as osp
import numpy as np
import unittest
from pyboreas.eval.localization import eval_local
from pyboreas.utils.utils import get_inverse_tf, se3ToSE3, rotation_error
from pyboreas.utils.odometry import read_traj_file_gt2
from pylgmath import so3op

# gt: 4 x 4 x N
# pred: 4 x 4
def get_matching_pose(pred, gt):
    return np.argmin((gt[0, 3, :] - pred[0, 3])**2 + \
        (gt[1, 3, :] - pred[1, 3])**2, axis=0)

def gen_fake_submission(gtpath, gt_ref_seq, ref, seq, predpath, dim=3):
    gt_ref_poses, gt_ref_times = read_traj_file_gt2(osp.join(gtpath, gt_ref_seq,
        'applanix', ref + '_poses.csv'), dim=dim)
    pred_poses, pred_times = read_traj_file_gt2(osp.join(gtpath,
        seq, 'applanix', ref + '_poses.csv'), dim=dim)
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


class LocalizationTestCase(unittest.TestCase):

    def test_fake_submission(self):
        pred = 'pyboreas/test/demo/pred/loc'
        if not osp.exists(pred):
            pred = '/tmp/'
        gt = 'pyboreas/test/demo/gt'
        ref_seq = 'boreas-2021-08-05-13-34'
        ref = 'lidar'
        seqs = ['boreas-2021-09-02-11-42']
        dim = 3
        radar = True if dim == 2 else False
        np.random.seed(42)
        for seq in seqs:
            # if not osp.exists(osp.join(pred, seq + '.txt')):
            gen_fake_submission(gt, ref_seq, ref, seq, pred, dim=dim)
        results = eval_local(pred, gt, seqs, ref_seq, radar=radar, plot_dir=pred)
        errs = results[0][0]
        trans_rmse_expected_m = np.sqrt(0.1)
        C = so3op.vec2rot(np.array([np.sqrt(0.01), np.sqrt(0.01), np.sqrt(0.01)]).reshape(3, 1))
        phi_rmse_expected_deg = rotation_error(C) * 180 / np.pi
        consistency_expected = 1.0
        self.assertTrue(np.abs(errs[0] - trans_rmse_expected_m) < 1e-2)
        self.assertTrue(np.abs(errs[1] - trans_rmse_expected_m) < 1e-2)
        self.assertTrue(np.abs(errs[2] - trans_rmse_expected_m) < 1e-2)
        self.assertTrue(np.abs(errs[3] - phi_rmse_expected_deg) < 3e-2)
        self.assertTrue(np.abs(errs[4] - consistency_expected) < 1e-2)


if __name__ == '__main__':
    unittest.main()
