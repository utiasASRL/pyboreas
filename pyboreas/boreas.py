import os.path as osp

from data_classes.splits import *
from data_classes.sequence import Sequence
from data_classes.sensors import Camera, Lidar, Radar
from vis.visualizer import BoreasVisualizer

class BoreasDataset:
	def __init__(self, root='/data/boreas/', split=odom_sample):
		self.root = root
		self.split = split
		self.camera_frames = []
		self.lidar_frames = []
		self.radar_frames = []
		self.sequences = []
		self.seqDict = {}  # seq string to index
		self.map = None  # TODO: Load the HD map data

		for seqSpec in split:
			seq = Sequence(root, seqSpec)
			self.sequences.append(seq)
			self.camera_frames += seq.camera_frames
			self.lidar_frames += seq.lidar_frames
			self.radar_frames += seq.radar_frames
			self.seqDict[seq.seqID] = len(self.sequences) - 1

	def get_seq_from_ID(self, seqID):
		return self.sequences[self.seqDict[seqID]]

	def get_seq(self, idx):
		return self.sequences[idx]

	def get_camera(self, idx):
		return self.camera_frames[idx]

	def get_lidar(self, idx):
		return self.lidar_frames[idx]

	def get_radar(self, idx):
		return self.radar_frames[idx]

	def get_sequence_visualizer(self, seqID):
		return BoreasVisualizer(self.get_seq_from_ID(seqID))

# These could either go here or in a separate file
# getGroundTruthOdometry(seqID)
# projectLidarOntoCamera(vis_options)
# projectBoundingBoxesOntoSensor(BB, Sensor) (include camera projection?)
# projectMapOntoView(position, orientation, extent)
# todo: convert a list of poses to benchmark format
# todo: provide example code for interpolating between poses (pysteam)
