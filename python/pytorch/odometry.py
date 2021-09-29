# TODO: clean this up, make it more generalizable, test it

import torch
import numpy as np
import cv2
import json
import argparse
from torch.utils.data import Dataset, DataLoader

from boreas import BoreasDataset
from data_classes.sequence import Sequence
from data_classes.splits import odom_sample, odom_train, odom_valid, odom_test
from utils.radar import load_radar, radar_polar_to_cartesian, mean_intensity_mask
from utils.utils import get_inverse_tf, get_transform3
from utils.odometry import computeKittiMetrics
from pytorch.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler

class BoreasOdomTorch(BoreasDataset, Dataset):
    """Boreas Radar Dataset"""
    def __init__(self, config, split=odom_sample):
        self.root = config['root']
        self.config = config
        self.split = split
        self.camera_frames = []
        self.lidar_frames = []
        self.radar_frames = []
        self.sequences = []
        self.seqDict = {}  # seq string to index
        self.map = None  # TODO: Load the HD map data

        drop = self.config['window_size'] - 1
        for seqSpec in split:
            seq = Sequence(root, seqSpec)
            seq.camera_frames = seq.camera_frames[:-drop]
            seq.lidar_frames = seq.lidar_frames[:-drop]
            seq.radar_frames = seq.radar_frames[:-drop]
            self.sequences.append(seq)
            self.camera_frames += seq.camera_frames
            self.lidar_frames += seq.lidar_frames
            self.radar_frames += seq.radar_frames
            self.seqDict[seq.seqID] = len(self.sequences) - 1

        if config['sensor_type'] == 'camera':
            self.frames = self.camera_frames
        elif config['sensor_type'] == 'lidar':
            self.frames = self.lidar_frames
        else:
            self.frames = self.radar_frames

    def __len__(self):
        return len(self.frames)

    def get_groundtruth_odometry(self, curr_frame, next_frame, make_se2=False):
        """Retrieves the groundtruth 4x4 transform from current time to next
        Args:
            curr_frame (SensorType): current frame object that contains sensor data, stamp, pose
            next_frame (SensorType)
        Returns:
            np.ndarray: 4x4 transformation matrix from current time to next (T_2_1)
        """

        # Note: frame.pose is T_enu_sensor
        T_2_1 = np.matmul(get_inverse_tf(next_frame.pose), curr_frame.pose)  # 4x4 SE(3)
        if make_se2:
            heading, _, _ = rotToYawPitchRoll(T_2_1[:3, :3])
            T_2_1 = get_transform3(T_2_1[0, 3], T_2_1[1, 3], heading)  # 4x4 SE(2)
        return T_2_1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.frames[idx]
        next_frame = self.frames[idx + 1]

        frame.load_data()

        out_dict = {}

        if config['sensor'] == 'camera':
            data = np.transpose(frame.img, (2, 0, 1))  # HWC --> CHW
            data = np.expand_dims(data.astype(np.float32), axis=0) / 255.0
        if config['sensor'] == 'radar':
            polar = np.expand_dims(frame.polar.astype(np.float32), axis=0) / 255.0
            data = np.expand_dims(frame.cartesian.astype(np.float32), axis=0) / 255.0
            mask = np.expand_dims(frame.mask.astype(np.float32), axis=0) / 255.0
            # polar_mask = mean_intensity_mask(polar)
            # data = frame.get_cartesian(self.config['cart_resolution'], self.config['cart_pixel_width'])
            # mask = frame.get_cartesian(self.config['cart_resolution'], self.config['cart_pixel_width'], polar_mask)
            out_dict['mask'] = mask
            out_dict['polar'] = polar
            out_dict['timestamps'] = np.expand_dims(frame.timestamps, axis=0)
            out_dict['azimuths'] = np.expand_dims(frame.azimuths, axis=0)
        if config['sensor'] == 'lidar':
            data = np.expand_dims(frame.points, axis=0)
            # TODO: voxelize pointcloud data
        out_dict['data'] = data

        t1 = frame.timestamp
        t2 = next_frame.timestamp
        t_ref = np.array([t1, t2]).reshape(1, 2)
        return {'data': data, 'T_2_1': T_2_1, 't_ref': t_ref}

def get_dataloaders(config):
    """Returns the dataloaders for training models in pytorch.
    Args:
        config (json): parsed configuration file
    Returns:
        train_loader (DataLoader)
        valid_loader (DataLoader)
        test_loader (DataLoader)
    """
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = BoreasOdomTorch(config, config['train_split'])
    valid_dataset = BoreasOdomTorch(vconfig, config['valid_split'])
    test_dataset = BoreasOdomTorch(vconfig, config['test_split'])
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pytorch/odom_config.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    config['train_split'] = odom_train
    config['valid_split'] = odom_valid
    config['test_split'] = odom_test

    train_loader, valid_loader, test_loader = get_dataloaders(config)

    model = YourModel(config).to(config['gpuid'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    ckpt_path = None
    if args.pretrain is not None:
        ckpt_path = args.pretrain

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device(config['gpuid']))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # training:
    model.train()
    for batchi, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch)
        if loss.requires_grad:
            loss.backward()
        optimizer.step()

    # TODO: write code for testing
    model.eval()
    t_errs = []
    r_errs = []
    for seqSpec in odom_test:
        config['test_split'] = seqSpec
        _, _, test_loader = get_dataloaders(config)
        T_gt = []
        T_pred = []
        for batchi, batch in enumerate(test_loader):
            with torch.no_grad()
                out = model(batch)
            T_gt.append(batch['T_2_1'][0].numpy().squeeze())
            T_pred.append(out['T_2_1'][0].numpy().squeeze())
        t_err, r_err = computeKittiMetrics(T_gt, T_pred, [len(T_gt)])
        t_errs.append(t_err)
        r_errs.append(r_err)
        print('KITTI t_err: {} %'.format(t_err))
        print('KITTI r_err: {} deg/m'.format(r_err))

    t_err = np.mean(t_errs)
    r_err = np.mean(r_errs)
    print('Average KITTI metrics over all test sequences:')
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))


