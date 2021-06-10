import argparse
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/boreas_dataset_root/', type=str,
                        help='location of root folder. Sequences are stored under root/')
    parser.add_argument('--sequence', default='boreas-2020-12-01-13-26', type=str,
                        help='name of the sequence to be visualized')
    args = parser.parse_args()
	
	# TODO: play back the sequence visualizing lidar, radar, camera data
	# No need to save or store each frame as a video, just display it.
	# Bonus points for enabling people to pause, rewind
