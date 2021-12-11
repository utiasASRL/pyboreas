import os
import os.path as osp
import numpy as np
from copy import deepcopy
# from shapely.geometry import Polygon

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train

# todo: generate fake detections (noise on gt + false positives + false negatives)

# def intersection_area(d, g):
# 	# top-down intersection area
# 	pd = Polygon(d.pc.points[:4, :2])
# 	pg = Polygon(g.pc.points[:4, :2])
# 	return pg.intersection(pd).area

# def boxOverlap(d, g, criterion=-1, dim=3):
# 	inter = intersection_area(d, g)
# 	if dim == 3:
# 		ymax = min(d.pos[2, 0], g.pos[2, 0])
# 		ymin = max(d.pos[2, 0] - d.extent[2, 0], g.pos[2, 0] - g.extent[2, 0])
# 		inter *= max(0, ymax - ymin)
# 	det = d.extent[0, 0] * d.extent[1, 0]
# 	gt = g.extent[0, 0] * g.extent[1, 0]
# 	if dim == 3:
# 		det *= d.extent[2, 0]
# 		gt *= g.extent[2, 0]
# 	if criterion == -1:  # union
# 		return inter / (det + gt - inter)
# 	elif criterion == 0:  # bbox_a
# 		return inter / det
# 	elif criterion == 1:  # bbox_b
# 		return inter / gt
# 	else:
# 		return 0

# def box3DOverlap(d, g, criterion)
# 	return boxOverlap(d, g, criterion, dim=3)

# def groundBoxOverlap(d, g, criterion)
# 	return boxOverlap(d, g, criterion, dim=2)

def convert_bbs_to_kitti(bbsIn):
    bbs = deepcopy(bbsIn)
    # kitti eval expects coordinates to be in a z-forwards, x-right, y-down frame.
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    for bb in bbs.bbs:
        np.array(annotations['rotation_y'].append(-1 * rotToYawPitchRoll(bb.rot)[0])).reshape(-1)
    T_camera_lidar = np.array([[0, -1, 0, 0],[0, 0, -1, 0],[1, 0, 0, 0],[0, 0, 0, 1]])
    bbs.transform(T_camera_lidar)
    # for bb in bbs.bbs:
    N = len(bbs.bbs)
    annotations['name'] = np.array([bb.label for bb in bbs.bbs])
    annotations['truncated'] = np.zeros(N)
    annotations['occluded'] = np.zeros(N)
    # annotations['alpha'] = np.array([np.arctan2(bb.pos[0, 0], bb.pos[2, 0]) for bb in bbs.bbs])
    annotations['alpha'] = -1 * np.ones(N)
    P = np.identity(4)  # orthographic projection
    UV = bbs.project(P, filterCamFront=False)
    bbox = []
    for uv in UV:
        xmin = np.min(uv[:, 0])
        xmax = np.max(uv[:, 0])
        ymin = np.min(uv[:, 1])
        ymax = np.max(uv[:, 1])
        bbox.append([xmin, ymin, xmax, ymax])
    annotations['bbox'] = np.array(bbox).reshape(-1, 4)
    assert(annotations['name'].shape[0] == annotations['bbox'].shape[0]), print(annotations['bbox'], annotations['name'])

    annotations['dimensions'] = np.array([[bb.extent[0, 0], bb.extent[2, 0], bb.extent[1, 0]] for bb in bbs.bbs]).reshape(-1, 3)
    annotations['location'] = np.array([[bb.pos[0, 0], bb.pos[1, 0], bb.pos[2, 0]] for bb in bbs.bbs]).reshape(-1, 3)

    has_score = True
    for bb in bbs.bbs:
        if bb.score is None:
            has_score = False
    if has_score:
        annotations['score'] = np.array([bb.score for bb in bbs.bbs])
    else:
        annotations['score'] = np.zeros(N)
    return annotations

def get_kitti_labels(root, labelFolder, split=obj_train):
    bd = BoreasDataset(root, split=split, labelFolder=labelFolder)
    annos = []
    # for lid in bd.lidar_frames:
    for i in range(100):
        lid = bd.lidar_frames[i]
        if not lid.has_bbs():
            continue
        bbs = lid.get_bounding_boxes()
        annos.append(convert_bbs_to_kitti(bbs))
    return annos

if __name__ == '__main__':
	# change obj_train to your own train/validation/test split
	bd_gt = BoreasDataset('/media/backup2/', split=obj_train, labelFolder='labels')
	bd_pred = BoreasDataset('/media/backup2/', split=obj_train, labelFolder='labels_pred')

	assert(len(bd_gt.lidar_frames) == len(bd_pred.lidar_frames))

	for lidgt, lidpred in zip(bd_gt.lidar_frames, bd_pred.lidar_frames):
		if not bd_gt.has_bbs():
			continue
		# associate predictions with ground truth
