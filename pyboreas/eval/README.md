# Benchmarks
## Odometry Benchmark
We offer an odometry benchmark for 3D (lidar, camera) and 2D (radar). Our evaluation metric is based on the KITTI odometry benchmark, which computes the relative position and orientation errors over segments of 100 m to 800 m in increments of 100 m.

### File format
Odometry results for each sequence are to be recorded into a separate `.txt` file and placed in the same directory. The name of each file is the sequence name followed by the `.txt` extension. The devkit contains an example for two sequences:
```
ls -1 pyboreas/test/demo/pred/3d/
boreas-2021-08-05-13-34.txt  
boreas-2021-09-02-11-42.txt
```

The format of each `.txt` file is a space separated N x 13 table, where N is the number of frames in the sequence, and each row corresponds to a single frame. The first column is the timestamp (64 bit integer) in microseconds. The remaining 12 columns are the upper 3 x 4 block of the SE(3) transformation matrix (64 bit float) in row-aligned order (i.e., the first 4 entries correspond to the first row).

Each row entry is an SE(3) transformation from a fixed reference frame to a local reference frame on the vehicle. E.g., a transformation `T_local_fixed` that transforms a homogenous point in the fixed frame `p_fixed` to the local frame:
```
p_local = T_local_fixed * p_fixed
```

The local frame is the `applanix` frame for the 3D benchmark, while for the 2D benchmark it is simply the `radar` sensor frame. The choice of the fixed reference frame does not matter as long as it is consistent for the entire sequence since the benchmark evaluates relatively. We suggest to simply set it as the first frame of the sequence.

### Local evaluation
The benchmark evaluation can be run locally for sequences with known groundtruth. For example, we will use the sequences under `pyboreas/test/demo/pred/3d/`.

From the root directory, go to the directory `pyboreas`:
```
cd pyboreas
```
We can see the arguments of the benchmark script as follows:
```
python eval/odometry_benchmark.py -h
usage: odometry_benchmark.py [-h] [--pred PRED] [--gt GT] [--radar] [--no-interp] [--no-solver]

optional arguments:
  -h, --help   show this help message and exit
  --pred PRED  path to prediction files
  --gt GT      path to groundtruth files
  --radar      evaluate radar odometry in SE(2)
  --no-interp  disable built-in interpolation
  --no-solver  disable solver for built-in interpolation
```