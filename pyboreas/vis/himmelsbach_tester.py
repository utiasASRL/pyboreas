from pyboreas.data.sequence import Sequence
from pyboreas.vis.visualizer import BoreasVisualizer
import time

seq = Sequence("/home/jqian/datasets/boreas-devkit/",["boreas_mini_v2", 1616518050000000, 1616518060000000])
#vis = BoreasVisualizer(seq)
#vis.visualize(0)

t = seq.lidar_frames[0]
t.load_data()
v = t.voxelize()
# t.random_downsample(0.7)
start = time.time()
total_load_time = 0
for i in range(5):
    load_start = time.time()
    t = seq.lidar_frames[i*10]
    t.load_data()
    total_load_time += time.time() - load_start
    t.remove_ground_himmelsbach(show=True)
    # t.voxelize(show=True)
#print(f"time: {time.time() - start}, avg {(time.time() - start)/10}, load: {total_load_time/10}")