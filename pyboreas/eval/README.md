# Benchmarks
## Odometry Benchmark
We offer an odometry benchmark for 3D (lidar, camera) and 2D (radar). Our evaluation metric is based on the KITTI odometry benchmark, which computes the relative position and orientation errors over segments of 100 m to 800 m with increments of 100 m.

### File format
Odometry results for each sequence are to be recorded into a separate `.txt` file and placed in the same directory. The name of each file is the sequence name followed by the `.txt` extension. The devkit contains an example demo with two sequences:
```
ls -1 pyboreas/test/demo/pred/3d/
boreas-2021-08-05-13-34.txt  
boreas-2021-09-02-11-42.txt
```

The format of each `.txt` file is a space separated `K x 13` table, where `K` is the total number of frames in the sequence, and each row `k` corresponds to the odometry estimate of frame `k`. The first column is the timestamp (64 bit integer) in microseconds. The remaining 12 columns are the upper 3 x 4 block of the SE(3) transformation matrix (64 bit float) in row-aligned order (i.e., the first 4 entries correspond to the first row).

The `k`th row entry is the SE(3) transformation between a stationary reference frame `i` and the `k`th estimated frame of the moving robot `vk`. E.g., a transformation `T_vk_i` that transforms a homogeneous point in the stationary frame `p_i` to the `k`th robot frame:
```
p_vk = T_vk_i * p_i
```
Set the robot frame `v` to be the `applanix` frame for the 3D benchmark, and to the `radar` sensor frame for the 2D benchmark. The choice of the stationary frame `i` does not matter as long as it is consistent for the entire sequence. We suggest to simply set it as the first frame of the sequence, `v0`. This will make the first row entry of the table `T_v0_v0` (i.e., the identity transformation), the second row will be `T_v1_v0`, the `k`th row will be `T_vk_v0`, and so on.

### Local evaluation
The benchmark evaluation can be run locally for sequences with known groundtruth. We will use the sequences under `pyboreas/test/demo/pred/3d/` for our example demo.

From the root directory, go to the directory `pyboreas`. We can see the arguments of the benchmark script as follows:
```
cd pyboreas
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
The `pred` argument is the directory containing the odometry sequence files, which is `pyboreas/test/demo/pred/3d/` for this demo. 

The `gt` argument is for the root directory of your dataset, which should contain the corresponding groundtruth files required for evaluation. For this demo we will use `pyboreas/test/demo/gt/`, but for your use case it should be the root directory of where you stored your data.

The `radar` argument should be included to evaluate the 2D benchmark. The 3D benchmark is run by default.

The `no-interp` argument should be included to evaluate without the built-in interpolation method. By default the benchmark will interpolate your odometry results to the timestamps of the evaluation groundtruth, which exactly correspond to the lidar frame timestamps for the 3D benchmark. The 2D benchmark timestamps exactly correspond to the radar timestamps. If the provided odometry estimates are provided with the correct timestamps, including `no-interp` will disable the interpolation and run much faster.

The `no-solver` argument should be included to disable the solver for the built-in interpolation method. The benchmark applies a batch optimization routine to solve for velocity estimates of each frame in order to interpolate. If the solver is disabled, the benchmark instead interpolates with velocities approximated with finite difference. This is less accurate, but will run faster.

This demo requires the built-in interpolation method. To run the demo without the solver:
```
python eval/odometry_benchmark.py --pred test/demo/pred/3d/ --gt test/demo/gt/ --no-solver
processing sequence boreas-2021-08-05-13-34.txt ...
boreas-2021-08-05-13-34.txt took 11.260696411132812  seconds
Error:  1.1093740051895995  %,  0.004420958778842962  deg/m 

processing sequence boreas-2021-09-02-11-42.txt ...
boreas-2021-09-02-11-42.txt took 26.46020483970642  seconds
Error:  0.02641292148342157  %,  0.00017168761982101885  deg/m 

Evaluated sequences:  ['boreas-2021-08-05-13-34.txt', 'boreas-2021-09-02-11-42.txt']
Overall error:  0.5678934633365106  %,  0.0022963231993319904  deg/m
```
To run the demo with the solver:
```
python eval/odometry_benchmark.py --pred test/demo/pred/3d/ --gt test/demo/gt/
```
Note that this can take a couple minutes for each sequence, depending on your hardware.

### Online evaluation
TODO
