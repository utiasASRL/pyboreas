import os.path as osp
import numpy as np
from pyboreas.utils.utils import get_time_from_filename, yawPitchRollToRot, rotToYawPitchRoll, get_transform2
from pyboreas.utils.utils import se3ToSE3, get_inverse_tf
from pyboreas.vis.vis_utils import draw_boxes
from pyboreas.data.pointcloud import PointCloud

class BoundingBoxes:
    def __init__(self):
        self.timestamp = None
        self.bbs = []
        self.path = None

    def load_from_file(self, path):
        assert(osp.exists(path))
        self.bbs = []
        self.path = path
        with open(path) as f:
            self.timestamp = get_time_from_filename(path)
            for line in f:
                parts = line.split()
                uuid = parts[0]
                label = parts[1]
                ext = np.array([float(parts[2]), float(parts[3]), float(parts[4])]).reshape(-3, 1) # TODO: fix
                pos = np.array([float(parts[5]), float(parts[6]), float(parts[7])]).reshape(-3, 1)
                yaw = float(parts[8])
                rot = yawPitchRollToRot(yaw, 0, 0)
                if parts[9] == 'None':
                    numPoints = 0
                else:
                    numPoints = int(parts[9])
                self.bbs.append(BoundingBox(pos, ext, rot, label, uuid, numPoints))

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for bb in self.bbs:
                yaw, _, _ = rotToYawPitchRoll(bb.rot)
                f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(bb.uuid,
                    bb.label, bb.extent[0, 0], bb.extent[1, 0], bb.extent[2, 0],
                    bb.pos[0, 0], bb.pos[1, 0], bb.pos[2, 0], yaw, bb.numPoints))

    def render_2d(self, ax, color='r', **kwargs):
        for bb in self.bbs:
            bb.render_2d(ax, color=color, **kwargs)

    def transform(self, T):
        for bb in self.bbs: 
            bb.transform(T)

    def remove_motion(self, body_rate, tref):
        for bb in self.bbs:
            bb.remove_motion(body_rate, tref)

    def project(self, P, width=2448, height=2048, checkdims=False):
        # assumes bounding boxes have already been transformed into the camera coordinates
        # does not modify points in place, returns list of np.array points (pixel coords for box corners)
        UV = []
        for bb in self.bbs:
            if bb.pos[2] < 0: # only keep bounding boxes in front of the camera
                continue
            uv = bb.project(P, width, height, checkdims)
            if uv is not None:
                UV.append(uv)
        return UV

    def visualize(self, img, P, width=2448, height=2048, checkdims=False, color=[0,255,0],
        line_width=2, draw_corners=False):
        UV = self.project(P, width, height, checkdims)
        draw_boxes(img, UV, color, line_width, draw_corners)

    def filter_empty(self):
        for i in range(len(self.bbs) - 1, -1, -1):
            if self.bbs[i].numPoints == 0:
                del self.bbs[i]

    def passthrough(self, bounds):
        # xmin, xmax, ymin, ymax, zmin, zmax
        for i in range(len(self.bbs) - 1, -1, -1):
            if self.bbs[i].pos[0, 0] < bounds[0] or \
                self.bbs[i].pos[0, 0] > bounds[1] or \
                self.bbs[i].pos[1, 0] < bounds[2] or \
                self.bbs[i].pos[1, 0] > bounds[3] or \
                self.bbs[i].pos[2, 0] < bounds[4] or \
                self.bbs[i].pos[2, 0] > bounds[5]:
                del self.bbs[i]

    def index_from_uuid(self, uuid):
        for i, bb in enumerate(self.bbs):
            if bb.uuid == uuid:
                return i
        return -1

    def interpolate(self, idx, timestamp, spose, seqLabelFiles, seqLabelTimes, seqLabelPoses):
        if seqLabelTimes[idx] < timestamp:
            lower = idx
            upper = idx + 1
        elif seqLabelTimes[idx] > timestamp:
            lower = idx - 1
            upper = idx
        bb2 = BoundingBoxes()
        self.load_from_file(seqLabelFiles[lower])
        bb2.load_from_file(seqLabelFiles[upper])
        T_enu_l1 = seqLabelPoses[lower]
        T_enu_l2 = seqLabelPoses[upper]
        T_l1_l2 = np.matmul(get_inverse_tf(T_enu_l1), T_enu_l2)
        bb2.transform(T_l1_l2)
        # Interpolation
        t1 = seqLabelTimes[lower]
        t2 = seqLabelTimes[upper]
        alpha1 = (t2 - timestamp) / (t2 - t1 + 1e-14)
        alpha2 = (timestamp - t1) / (t2 - t1 + 1e-14)
        self._interpolate(alpha1, bb2, alpha2)
        T = np.matmul(get_inverse_tf(spose), T_enu_l1)
        self.transform(T)

    # may delete non-overlapping boxes 
    def _interpolate(self, alpha1, bb2, alpha2):
        for i in range(len(self.bbs) - 1, -1, -1):
            idx = bb2.index_from_uuid(self.bbs[i].uuid)
            if idx < 0:
                del self.bbs[i]
                continue
            self.bbs[i]._interpolate(alpha1, bb2.bbs[idx], alpha2)

class BoundingBox:
    def __init__(self, position=np.zeros((3, 1)), extent=np.zeros((3, 1)),
        rotation=np.identity(3), label=None, uuid=None, numPoints=None):
        """Checks dimensional consistency of inputs and constructs points array

        Args:
            position: (x,y,z) position of bbox centroid
            extent: (width, length, height) of the bbox
            rotation: rotation matrix for bbox orientation
            label: bounding box label (class)
            uuid: unique ID for a bounding box track
            numPoints: number of lidar points associated with this bounding box
        """
        assert(position.shape[0] == 3 and position.shape[1] == 1)
        assert(extent.shape[0] == 3 and extent.shape[1] == 1)
        assert(rotation.shape[0] == 3 and rotation.shape[1] == 3)
        self.pos = position
        self.extent = extent
        self.rot = rotation
        self.label = label
        self.uuid = uuid
        self.numPoints = numPoints
        self.timestamp = None

        # Construct array to extract points from extent
        # self.corner_map = {'ftr':0, 'ftl':1, 'btl':2, 'btr':3,
        #               'fbr':4, 'fbl':5, 'bbl':6, 'bbr':7}
        dims = [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]

        def _get_point_with_offset(pose, offset):
            p = np.array([offset[0],offset[1],offset[2], 1]).reshape(-1, 1)
            return np.matmul(pose, p)[:3, 0]

        pose = get_transform2(self.rot, self.pos)
        points = []
        for i in range(len(dims)):
            points.append(_get_point_with_offset(pose, self.extent.squeeze() * np.array(dims[i]) / 2))
        self.pc = PointCloud(np.array(points))

    def render_2d(self, ax, color="r", **kwargs):
        """Render the bbox into a top-down 2d view

        Args:
            ax: the axis to render the bbox onto
        """
        prev_pt = self.pc.points[3, :]
        for i in range(4):  # Just draw top 4 points of bbox
            ax.plot([prev_pt[0], self.pc.points[i, 0]], [prev_pt[1], self.pc.points[i, 1]], color=color, **kwargs)
            prev_pt = self.pc.points[i, :]

    def project(self, P, width=2448, height=2048, checkdims=False) -> np.ndarray:
        """
        Project bounding boxes corners onto 2D image plane using camera matrix P
        """

        uv, _, _ = self.pc.project_onto_image(P=P, width=width, height=height, checkdims=checkdims)
        if checkdims and uv.shape[0] < self.pc.points.shape[0]:
            return None

        return uv

    def transform(self, T):
        assert(T.shape[0] == 4 and T.shape[1] == 4)
        pose = get_transform2(self.rot, self.pos)
        pose = np.matmul(T, pose)
        self.pos = pose[:3, 3:]
        self.rot = pose[:3, :3]
        self.pc.transform(T)

    def remove_motion(self, body_rate, tref):
        # tref should be set to max time from associated pointcloud
        assert(self.timestamp is not None)
        T_undistort = se3ToSE3((self.timestamp - tref) * body_rate)
        self.transform(T_undistort)

    def _interpolate(self, alpha1, b2, alpha2):
        # linear interpolation
        self.pos = alpha1 * self.pos + alpha2 * b2.pos
        y1, p1, r1 = rotToYawPitchRoll(self.rot)
        y2, p2, r2 = rotToYawPitchRoll(b2.rot)
        y = alpha1 * y1 + alpha2 * y2
        p = alpha1 * p1 + alpha2 * p2
        r = alpha1 * r1 + alpha2 * r2
        self.rot = yawPitchRollToRot(y, p, r)
        self.pc.points = alpha1 * self.pc.points + alpha2 * b2.pc.points