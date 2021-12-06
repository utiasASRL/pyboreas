# Benchmarks
## Odometry Benchmark
We offer an odometry benchmark for 3D (lidar, camera) and 2D (radar). Our evaluation metric is based on the KITTI odometry benchmark, which computes the relative position and orientation errors over segments of 100 m to 800 m with increments of 100 m.

### File Format
Odometry results for each sequence are to be recorded into a separate `.txt` file and placed in the same directory. The name of each file is the sequence name followed by the `.txt` extension. The devkit contains an example demo with two sequences:
```
ls -1 pyboreas/test/demo/pred/3d/
boreas-2021-08-05-13-34.txt  
boreas-2021-09-02-11-42.txt
```

The format of each `.txt` file is a space separated `K x 13` table, where `K` is the total number of frames in the sequence, and each row `k` corresponds to the odometry estimate of frame `k`. The first column is the timestamp (64 bit integer) in microseconds. The remaining 12 columns are the upper 3 x 4 block of the SE(3) transformation matrix (64 bit float) in row-aligned order (i.e., the first 4 entries correspond to the first row).

The `k`th row entry is the SE(3) transformation between a stationary reference frame `i` and the `k`th estimated frame of the moving robot. E.g., a transformation `T_k_i` that transforms a homogeneous point in the stationary frame `p_i` to the `k`th robot frame:
```
p_k = T_k_i * p_i
```
The moving robot frame and corresponding timestamps for evaluation depend on whether you are submitting to the 3D benchmark or 2D benchmark.

### Evaluation Frame and Timestamps (3D Benchmark)
Set the moving robot frame to be the `applanix` frame for the 3D benchmark. The choice of the stationary frame `i` does not matter as long as it is consistent for the entire sequence. We suggest to simply set it as the first frame of the sequence. This will make the first row entry of the table `T_applanixk_applanix0` (i.e., the identity transformation), the second row will be `T_applanix1_applanix0`, the `k`th row will be `T_applanixk_applanix0`, and so on.

For the 3D benchmark, provide odometry estimates that **correspond exactly to the lidar sensor timestamps**. 

#### Example Scenario 1: Lidar Odometry
Report SE(3) pose estimates corresponding to each lidar sensor frame. If your estimator outputs estimates in the lidar sensor frame, make sure you transform your estimates to the `applanix` frame, e.g.,
```
T_applanixk_applanix0 = T_applanix_lidar * T_lidark_lidar0 * T_lidar_applanix
```
where `T_applanix_lidar` is the extrinsic calibration between the `applanix` frame and `lidar` sensor frame.

#### Example Scenario 2: Visual Odometry (Camera)
You will need to interpolate your pose estimates to match the lidar timestamps. Either use an interpolation method of your choice, or apply the interpolation method offered in this devkit. A demonstration is shown in subsection `Interpolation`.

Similar to [`Example Scenario 1: Lidar Odometry`](#example-scenario-1:-lidar-odometry), make sure to transform your `camera` frame estimates to the `applanix` frame, e.g.,
```
T_applanixk_applanix0 = T_applanix_camera * T_camerak_camera0 * T_camera_applanix
```
where `T_applanix_camera` is the extrinsic calibration between the `applanix` frame and `camera` sensor frame.

### Evaluation Frame and Timestamps (2D Benchmark)
Set the moving robot frame to be the `radar` sensor frame for the 2D benchmark. In other words, nothing needs to be done if your estimator is already reporting results in the `radar` frame. Your odometry estimates will be SE(2) poses, but still report your results as the corresponding SE(3) matrix.

Provide odometry estimates that **correspond exactly to the radar sensor timestamps**.

### Local Evaluation
The benchmark evaluation can be run locally for sequences with known groundtruth. We will use the sequences under `pyboreas/test/demo/pred/3d/` for our example demo.

From the root directory, go to the directory `pyboreas`. We can see the arguments of the benchmark script as follows:
```
cd pyboreas
python eval/odometry.py -h
usage: odometry.py [-h] [--pred PRED] [--gt GT] [--radar] [--no-interp] [--no-solver]

optional arguments:
  -h, --help            show this help message and exit
  --pred PRED           path to prediction files
  --gt GT               path to groundtruth files
  --radar               evaluate radar odometry in SE(2)
  --interp INTERP       path to interpolation output, do not set if evaluating
  --processes PROCESSES
                        number of workers to use for built-in interpolation
  --no-solver           disable solver for built-in interpolation
```
The `pred` argument is the directory containing the odometry sequence files, which is `pyboreas/test/demo/pred/3d/` for this demo. 

The `gt` argument is for the root directory of your dataset, which should contain the corresponding groundtruth files required for evaluation. For this demo we will use `pyboreas/test/demo/gt/`, but for your use case it should be the root directory of where you stored your data.

The `radar` argument should be included to evaluate the 2D benchmark. The 3D benchmark is run by default.

The `interp` argument should be set as the output directory for the interpolation files. Setting this argument will change the operation of the benchmark script to interpolation mode, i.e., it will not compute the errors and output error results. Instead, it will output a `.txt` file for each of your odometry sequences interpolated at the groundtruth (lidar) timestamps. Use this argument only if you need to interpolate.

The `processes` argument sets the number of processes to use when interpolating. Setting this argument to 1 will result in no additional subprocesses being created. Default value is the CPU count of your machine.

The `no-solver` argument should be included to disable the solver for the built-in interpolation method. We use a batch optimization routine to solve for velocity estimates of each frame in order to interpolate. If the solver is disabled, the script instead interpolates with velocities approximated with finite difference. This is less accurate, but will run much faster. Suggested use is for debugging.

This demo requires the built-in interpolation method. We will interpolate without the solver just for demonstration purposes (run it faster):
```
python eval/odometry.py --pred test/demo/pred/3d/ --gt test/demo/gt/ --interp test/demo/pred/3d/interp/ --processes 1 --no-solver

interpolating sequence boreas-2021-08-05-13-34.txt ...
boreas-2021-08-05-13-34.txt took 9.404045581817627  seconds
output file: test/demo/pred/3d/interp/boreas-2021-08-05-13-34.txt 

interpolating sequence boreas-2021-09-02-11-42.txt ...
boreas-2021-09-02-11-42.txt took 24.32158589363098  seconds
output file: test/demo/pred/3d/interp/boreas-2021-09-02-11-42.txt
```

Now to evaluate the interpolated sequences:
```
python eval/odometry.py --pred test/demo/pred/3d/interp/ --gt test/demo/gt/

processing sequence boreas-2021-08-05-13-34.txt ...
boreas-2021-08-05-13-34.txt took 2.0557074546813965  seconds
Error:  1.1093740051895995  %,  0.004420958778842962  deg/m 

processing sequence boreas-2021-09-02-11-42.txt ...
boreas-2021-09-02-11-42.txt took 1.6580510139465332  seconds
Error:  0.02641292148342157  %,  0.00017168761982101885  deg/m 

Evaluated sequences:  ['boreas-2021-08-05-13-34.txt', 'boreas-2021-09-02-11-42.txt']
Overall error:  0.5678934633365106  %,  0.0022963231993319904  deg/m
```

### Online Evaluation
TODO
