import os
import os.path as osp

from pyboreas.data.splits import *
from pyboreas.data.sequence import Sequence
from pyboreas.data.sensors import Camera, Lidar, Radar
from pyboreas.vis.visualizer import BoreasVisualizer

class BoreasDataset:
	def __init__(self, root='/data/boreas/', split=None):
		self.root = root
		self.split = split
		self.camera_frames = []
		self.lidar_frames = []
		self.radar_frames = []
		self.sequences = []
		self.seqDict = {}  # seq string to index
		self.map = None  # TODO: Load the HD map data

		if split is None:
			split = sorted([[f] for f in os.listdir(root) if 'boreas' in f])

		for seqSpec in split:
			seq = Sequence(root, seqSpec)
			self.sequences.append(seq)
			self.camera_frames += seq.camera_frames
			self.lidar_frames += seq.lidar_frames
			self.radar_frames += seq.radar_frames
			self.seqDict[seq.ID] = len(self.sequences) - 1

	def get_seq_from_ID(self, ID):
		return self.sequences[self.seqDict[ID]]

	def get_seq(self, idx):
		return self.sequences[idx]

	def get_camera(self, idx):
		self.camera_frames[idx].load_data()
		return self.camera_frames[idx]

	def get_lidar(self, idx):
		self.lidar_frames[idx].load_data()
		return self.lidar_frames[idx]

	def get_radar(self, idx):
		self.radar_frames[idx].load_data()
		return self.radar_frames[idx]

	def get_sequence_visualizer(self, ID):
		return BoreasVisualizer(self.get_seq_from_ID(ID))

# These could either go here or in a separate file
# getGroundTruthOdometry(ID)
# projectLidarOntoCamera(vis_options)
# projectBoundingBoxesOntoSensor(BB, Sensor) (include camera projection?)
# projectMapOntoView(position, orientation, extent)
# TODO: convert a list of poses to benchmark format
# TODO: provide example code for interpolating between poses (pysteam)
