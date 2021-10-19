# TODO: plot odometry results vs. ground truth

import sys
import glob
import csv

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from tqdm import tqdm

from pyboreas.vis.boreas_plotter import BoreasPlotter
from pyboreas.data.sequence import Sequence
from pyboreas.utils.utils import get_closest_index

# matplotlib.use("tkagg")  # slu: for testing with ide


def get_closest_frame(query_time, target_times, targets):
    closest = get_closest_index(query_time, target_times)
    assert(np.abs(query_time - target_times[closest]) < 1.0), "closest to query: {} not found.".format(query_time)
    return targets[closest]


class BoreasVisualizer:
    """Main class for loading the Boreas dataset for visualization.

    Loads relevant data, transforms, and labels and provides access to several visualization options.
    Currently only works for one track at a time.
    """

    def __init__(self, sequence):
        """Initialize the class with the corresponding data, transforms and labels.

        Args:
            sequence: Sequence object to base the visualization on
        """
        self.sequence = sequence
        self.calib = sequence.calib
        self.track_length = len(sequence.lidar_frames)
        self.lidar_frames = sequence.lidar_frames
        lstamps = [frame.timestamp for frame in sequence.lidar_frames]
        cstamps = [frame.timestamp for frame in sequence.camera_frames]
        rstamps = [frame.timestamp for frame in sequence.radar_frames]
        # Get corresponding camera and radar frame for each lidar frame
        self.camera_frames = [get_closest_frame(lstamp, cstamps, sequence.camera_frames) for lstamp in lstamps]
        self.radar_frames = [get_closest_frame(lstamp, rstamps, sequence.radar_frames) for lstamp in lstamps]

        # # Load label data
        # print("Loading Labels...", flush=True)
        # with open(self.label_file, 'r') as file:
        #     raw_labels = json.load(file)
        #     for label in tqdm(raw_labels):
        #         self.labels.append(label['cuboids'])

    def visualize(self, frame_idx, predictions=None, mode='both', show=True):
        """
        Visualize the sequence.

        Args:
            frame_idx: the frame index in the current sequence to visualize
            predictions: user generated predictions to also be visualized
            mode: 'persp' for perspective (camera) view, 'bev' for top down view, or 'both'
            show: flag to show the matplotlib plot. Should only be off for exporting to video

        Returns: the BoreasPlotter object used for visualizing

        """
        boreas_plot = BoreasPlotter(self, frame_idx, mode=mode)
        boreas_plot.update_plots(frame_idx)
        if show:
            plt.show()
        else:
            plt.close(boreas_plot.fig)

        return boreas_plot

    def export_vis_video(self, name, mode='both'):
        """
        Exports the current sequence visualization to video. Defaults to current working directory for save location.

        Args:
            name: name for the exported video file (don't include extension)
            mode: 'persp' for perspective (camera) view, 'bev' for top down view, or 'both'
        """
        imgs = []
        # Render the matplotlib figs to images
        print("Exporting Visualization to Video")
        for i in tqdm(range(len(self.lidar_frames)), file=sys.stdout):
            bplot = self.visualize(frame_idx=i, mode=mode, show=False)
            canvas = FigureCanvas(bplot.fig)
            canvas.draw()
            graph_image = np.array(bplot.fig.canvas.get_renderer()._renderer)
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            imgs.append(graph_image)

        # Write the images to video
        # MJPG encoder is part of cv2
        out = cv2.VideoWriter(name + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, (graph_image.shape[1], graph_image.shape[0]))
        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()


if __name__ == '__main__':
    sequence = Sequence("/home/shichen/datasets/", ["boreas_mini_v2", 1616518050000000, 1616518060000000])
    dataset = BoreasVisualizer(sequence)
    # for name in ["both", "persp", "bev"]:
    #     dataset.export_vis_video(name, name)
    dataset.visualize(50)
