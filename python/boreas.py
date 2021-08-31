from splits import *
from sequence import Sequence
from sensors import Camera, Lidar, Radar

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