The purpose of our metric localization leaderboard is to benchmark mapping and localization pipelines. In this scenario, we envision a situation where one or more repeated traversals of the Glen Shields route are used to construct a map offline. Any and all data  from the training sequences may be used to construct a map in any fashion.

Then, during a test sequence, the goal is to perform metric localization between the live sensor data and the pre-built map. Localization approaches may make use of temporal filtering and can leverage the IMU if desired but GPS information will not be available. The goal of this benchmark is to simulate localizing a vehicle in real-time and as such methods may not use future sensor information in an acausal manner.

Our goal is to support both global and relative map structures. Only one of the training sequences will specified as the map sequence used by the benchmark. For 3D localization, users must choose either the lidar or camera as the reference sensor. For 2D localization, only the radar frames are used as a reference. For each (camera|lidar|radar) frame `s_2` in the test sequence, users will specify the ID (timestamp) of the (camera|lidar|radar) frame `s_1` in the map sequence that they are providing a relative pose with respect to: `pred_T_s1_s2`. We then compute root-mean squared error (RMSE) values for the translation and rotation as follows:

```Python
T_as = T_applanix_sensor
T_sa = get_inverse_tf(T_as)
T = pred_T_s1_s2 @ get_inverse_tf(gt_T_s1_s2)
Te = T_as @ T @ T_sa
xe, ye, ze = Te[:3, 3]
phie = np.arccos(np.clip(0.5 * (np.trace(Te[:3, :3]) - 1), -1, 1))
```
For each test sequence, users will provide a space-seperated text file of K x (14|50) values with the same name as the test sequence, example: `boreas-2021-11-28-09-18.txt`. The first column is the timestamp of the test frame, the second column is the timestamp of the user-specified reference frame in the map sequence. The following 12 values correspond to the upper 3x4 component of `pred_T_s1_s2` in row-major order. Users also have the option of providing 6x6 inverse covariance matrices `Sigma_inv_s1_s2` for each localization estimate. The entire covariance matrix must be unrolled into 36 values (row-major order) and appended to each row, for a total of 50 values per row. We will use these covariance values to calculate consistency scores `c`:

```Python
xi = SE3Tose3(T)
c = xi.T @ Sigma_inv_s1_s2 @ xi
c = np.sqrt(c / 6)
```

For the localization benchmark, an additional metadata tag is required: `reference` which can be one of: `lidar, radar, camera`. This is used to determine which sensor is being used as a reference for ground truth poses.
