import os

import numpy as np
import cv2
import yaml

from splits import *
from utils.utils import carrot, get_transform, load_lidar
from utils.radar import load_radar, radar_polar_to_cartesian

class PointCloud:
	def __init__(self, points):
		# x, y, z, (i, r, t)
		self.points = points

	def transform(self, T):
		assert(T.shape[0] == 4 and T.shape[1] == 4)
		for i in range(self.points.shape[0]):
			pbar = np.vstack((self.points[i, :3], np.array([1])))
			pbar = np.matmul(T, pbar)
			self.points[i, :3] = pbar[:3, 0]
	
	def remove_motion(self, points, body_rate, tref=None, in_place=True):
		# body_rate: (6, 1) [vx, vy, vz, wx, wy, wz] in body frame
		# Note: modifies points contained in this class
		assert(body_rate.shape[0] == 6 and body_rate.shape[1] == 1)
		tmin = np.min(points[:, 5])
		tmax = np.max(points[:, 5])
		if tref is None:
			tref = (tmin + tmax) / 2
		# Precompute transforms for compute speed
		bins = 101
		delta = (tmax - tmin) / (bins - 1)
		T_undistorts = []
		for i in range(bins):
			t = tmin + i * delta
			T_undistorts.append((t - tref) * carrot(body_rate))
		if not in_place:
			ptemp = np.copy(points)
		for i in range(self.points.shape[0]):
			pbar = np.vstack((self.points[i, :3], np.array([1])))
			index = int((self.points[i, 5] - tmin) / delta)
			pbar = np.matmul(T_undistorts[index], pbar)
			if in_place:
				self.points[i, :3] = pbar[:3, 0]
			else:
				ptemp[i, :3] = pbar[:3, 0]
		if not in_place:
			return ptemp

	# TODO: remove_ground(self, bool: in_place)

class Sensor:
	def __init__(self, path):
		self.path = path
		self.frame = path.split('/')[-1]
		self.sensType = path.split('/')[-2]
		self.seqID = path.split('/')[-3]
		self.root = '/'.join(path.split('/')[:-2] + [''])
		self.pose = None
		self.velocity = None
		self.body_rate = None
		self.timestamp = None

	def get_pose(self):
		if self.pose is not None:
			return self.pose
		posepath = self.root + 'applanix/' + self.sensType + '_poses.csv'
		with open(posepath, 'r') as f:
			f.readline()  # header
			for line in f:
				if line.split(',')[0] == self.frame:
					gt = [float(x) for x in line.split(',')]
					self.pose = get_transform(gt)
					wbar = np.array([gt[12], gt[11], gt[10]]).reshape(3, 1)
					wbar = np.matmul(self.pose[:3, :3], wbar)
					self.velocity = np.array([gt[4], gt[5], gt[6], wbar[0], wbar[1], wbar[2]]).reshape(6, 1)
					vbar = np.array([gt[4], gt[5], gt[6]]).reshape(3, 1)
					vbar = np.matmul(self.pose[:3, :3].T, vbar)
					self.body_rate = np.array([vbar[0], vbar[1], vbar[2], gt[12], gt[11], gt[10]])
					self.timestamp = gt[1]
					return self.pose

	def get_velocity(self):
		if self.velocity is None:
			self.get_pose()
		return self.velocity

	def get_body_rate(self):
		if self.body_rate is None:
			self.get_pose()
		return self.body_rate

	def get_timestamp(self):
		if self.timestamp is None:
			self.get_pose()
		return self.timestamp

class Lidar(Sensor, PointCloud):
	def __init__(self, path):
		Sensor.__init__(self, path)
		self.points = load_lidar(path)
		self.timestamp = None
	
	# TODO: get_bounding_boxes()
	# TODO: get_semantics()
	# TODO: visualize(int: projection, bool: use_boxes)

class Camera(Sensor):
	def __init__(self, path):
		Sensor.__init__(self, path)
		self.img = cv2.imread(path)

	# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
	# TODO: get_semantics() # retrieve from file, cache to class variable
	# TODO: visualize(bool: use_boxes, Lidar: points (optional_arg))

class Radar(Sensor):
	def __init__(self, path):
		Sensor.__init__(self, path)
		self.timestamps, self.azimuths, _, self.polar = load_radar(path)

	def get_cartesian(self, radar_resolution, cart_resolution, cart_pixel_width):
		return radar_polar_to_cartesian(self.azimuths, self.polar, radar_resolution,
										cart_resolution)

	# TODO: get_bounding_boxes() # retrieve from file, cache to class variable
	# TODO: visualize(bool: use_boxes)

class Calib:
	def __init__(self, path):
		self.P0 = np.loadtxt(path + 'P_camera.txt')
		self.T_applanix_lidar = np.loadtxt(path + 'T_applanix_lidar.txt')
		self.T_camera_lidar = np.loadtxt(path + 'T_camera_lidar.txt')
		self.T_radar_lidar = np.loadtxt(path + 'T_radar_lidar.txt')
		
class Sequence:
	def __init__(self, root, seqSpec):
		self.seqID = seqSpec[0]
		self.path = root + self.seqID
		self.start = str(seqSpec[1])
		self.end = str(seqSpec[2])
		self.cameraFrames = os.listdir(self.path + '/camera/')
		self.lidarFrames = os.listdir(self.path + '/lidar/')
		self.radarFrames = os.listdir(self.path + '/radar/')
		self.cameraFrames = [self.path + '/camera/' + f for f in self.cameraFrames if self.start <= f and f <= self.end]
		self.lidarFrames = [self.path + '/lidar/' + f for f in self.lidarFrames if self.start <= f and f <= self.end]
		self.radarFrames = [self.path + '/lidar/' + f for f in self.radarFrames if self.start <= f and f <= self.end]
		self.cameraFrames.sort()
		self.lidarFrames.sort()
		self.radarFrames.sort()
		self.calib = Calib(root + self.seqID + '/calib/')
		# TODO: load printable metadata string

	@property
	def cam0(self):
		for f in self.cameraFrames:
			yield Camera(f)

	def get_camera(self, idx):
		return Camera(self.cameraFrames[idx])

	@property
	def lidar(self):
		for f in self.lidarFrames:
			yield Lidar(f)

	def get_lidar(self, idx):
		return Lidar(self.lidarFrames[idx])

	@property
	def radar(self):
		for f in self.radarFrames:
			yield Radar(f)

	def get_radar(self, idx):
		return Radar(self.radarFrames[idx])

	def visualize(self):
		pass
		# TODO: generate video for the entire sequences
		# option 1: display video, option 2: save video to file

	def get_pose(self, sensType, timestamp):
		pass
		# TODO

class BoundingBoxes:
	def __init__(self, path):
		# pose / sensor frame
		# timestamp
		# 3D
		# BBs[]
		pass
	# load/save from/to file

#class BoundingBox2D: TODO
#class BoundingBox3D: TODO

class BoreasDataset:
	def __init__(self, root='/data/boreas/', split=odom_sample):
		self.root = root
		self.split = split
		self.cameraFrames = []
		self.lidarFrames = []
		self.radarFrames = []
		self.sequences = []
		self.seqDict = {}  # seq string to index
		self.map = None  # TODO: Load the HD map data

		for seqSpec in split:
			seq = Sequence(root, seqSpec)
			self.sequences.append(seq)
			self.cameraFrames += seq.cameraFrames
			self.lidarFrames += seq.lidarFrames
			self.radarFrames += seq.radarFrames
			self.seqDict[seq.seqID] = len(self.sequences) - 1

	def get_seq_from_ID(self, seqID):
		return self.sequences[self.seqDict[seqID]]

	@property
	def cam0(self):
		for f in self.cameraFrames:
			yield Camera(f)

	def get_camera(self, idx):
		return Camera(self.cameraFrames[idx])

	@property
	def lidar(self):
		for f in self.lidarFrames:
			yield Lidar(f)

	def get_lidar(self, idx):
		return Lidar(self.lidarFrames[idx])

	@property
	def radar(self):
		for f in self.radarFrames:
			yield Radar(f)

	def get_radar(self, idx):
		return Radar(self.radarFrames[idx])

# These could either go here or in a separate file
# getGroundTruthOdometry(seqID)
# projectLidarOntoCamera(vis_options)
# projectBoundingBoxesOntoSensor(BB, Sensor) (include camera projection?)
# projectMapOntoView(position, orientation, extent)
# todo: convert a list of poses to benchmark format
# todo: provide example code for interpolating between poses (pysteam)