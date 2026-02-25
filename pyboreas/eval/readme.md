# Evaluation

Submissions are `.zip` files which contain the result `.txt` files along with a `metadata.yaml` file.

Note that train/test splits are contained in `pyboreas.data.splits.py`
Users are free to create their own train/validation splits from the available training data. 

### Boreas Dataset Benchmark Sequences

All sequences below are the **odometry benchmarks**. 
For **localization**, use `boreas-2020-11-26-13-58` as the mapping sequence and submit results only for sequences marked with ✅.

| Sequence | Localization|
|----------|-------------|
| boreas-2020-12-04-14-00 | ✅ |
| boreas-2021-01-26-10-59 | ✅ |
| boreas-2021-02-09-12-55 | ✅ |
| boreas-2021-03-09-14-23 | ✅ |
| boreas-2021-04-22-15-00 | ❌ |
| boreas-2021-06-29-18-53 | ✅ |
| boreas-2021-06-29-20-43 | ❌ |
| boreas-2021-09-08-21-00 | ✅ |
| boreas-2021-09-09-15-28 | ❌ |
| boreas-2021-10-05-15-35 | ✅ |
| boreas-2021-10-26-12-35 | ✅ |
| boreas-2021-11-06-18-55 | ❌ |
| boreas-2021-11-28-09-18 | ✅ |

### Boreas Road Trip Dataset Benchmark Sequences

All sequences below are the **odometry benchmarks**. 
For **localization**, use the first sequence of each route as the mapping sequence and submit results only for sequences marked with ✅.

| Route            | Sequence               | Localization |
|-----------------|----------------------|-------------|
| Suburbs Route    | boreas-2024-12-03-12-54 | ❌ (use for mapping)|
| Suburbs Route    | boreas-2025-01-08-11-22 | ✅ |
|  Suburbs Route   | boreas-2025-02-15-17-19 | ✅ |
| Industrial Route | boreas-2024-12-05-14-12 | ❌ (use for mapping) |
| Industrial Route | boreas-2024-12-23-16-27 | ✅ |
| Industrial Route | boreas-2024-12-23-16-44 | ✅ |
| Skyway Route     | boreas-2024-12-04-11-45 | ❌ (use for mapping) |
| Skyway Route     | boreas-2024-12-04-12-08 | ✅ |
| Skyway Route     | boreas-2024-12-04-12-34 | ✅ |
| Tunnel East      | boreas-2024-12-04-14-28 | ❌ (use for mapping) |
| Tunnel East      | boreas-2024-12-04-14-50 | ✅ |
| Tunnel East      | boreas-2024-12-04-15-19 | ✅ |
| Farm             | boreas-2025-07-18-14-55 | ❌ (use for mapping) |
| Farm             | boreas-2025-07-18-15-30 | ✅ |
| Farm             | boreas-2025-07-18-16-05 | ✅ |


### Submission Formatting

The name of the `.zip` file must start with the desired dataset, which is one of `[boreas, boreasrt]`, then contain your method name, and end in the desired benchmark, which can be one of `[odometry, localization, detection]`. Example: `boreas-RoBoDoMeTrY-odometry.zip`. The dataset and benchmark specified in the file name must match their respecive fields specified in the yaml file.

metadata.yaml uses the following format:

```YAML
# options: [odometry, localization, detection]
benchmark: odometry

# options: [boreas, boreasrt]
dataset: boreas

# free text
methodname: RoBoDoMeTrY

# the email used for your account/login
email: your.email@example.com

# bool
2d: False

# free text
author: First Last, First1 Last1
papertitle: Our Wondrous Algorithm

# link to paper ("N/A" if not applicable)
paperurl: https://www.website/paperURL

# conference or journal name ("N/A" if not applicable)
venue: ICRA

# 4-digit year
year: 2077

# compute per frame in seconds (number)
runtimeseconds: 0.1

# computer specs (run `lscpu` in terminal and copy the output of "Model name")
computer: Intel i7-1370p

# sensor options for odometry: ['lidar', 'radar', 'IMU', 'camera']
sensors: ['lidar', 'IMU']

# sensor options for localization: ['lidar', 'radar', 'camera']
ref_sensor: ['lidar']
test_sensor: ['lidar']
```

- `benchmark` can be one of: `[odometry, localization, detection]`
- `dataset` can be one of: `[boreas, boreasrt]`
- `methodname` is a short nickname for your method
- `email` must match the email used to create your Boreas account
- `2d` if True, evaluate using SE(2) or BEV for object detections, otherwise submissions are evaluated in SE(3) or 3D for object detections
- `author` is a comma-separated list of author names
- `author, papertitle, paperurl, venue` are optional tags which can be left blank for an anonymous submission and may be updated via the website later

For the localization benchmark, two additional metadata tags are required: 
`ref_sensor` which can be one of: `lidar, radar, camera`. This is used to determine which sensor is being used as a reference for ground truth poses.
`test_sensor` which can be one of: `lidar, radar, camera`. This is used to determine which sensor acts as the "test" sensor.

<u>FMCW lidar results are not supported. Submit Velodyne lidar results only.</u>

### Submission Management

To create your submission `.zip`, put `metadata.yaml` and all of your result `.txt` files in the same directory, select them all, then compress them into a single zip archive (ensure the zip contains the files at its root, not an extra top-level folder).

We provide a Python file to check the format of the submission `.zip` file: `pyboreas.eval.submission_checker.py`. Please use this script to check your submission before uploading to the website. Your submission may fail silently if this has not been done.

See the other readme files for the format of the submission `.txt` files.

Once your submission has been uploaded, it will take several minutes to process your submission. When processing is complete, a confirmation will be sent to the `email` provided in the yaml file. <u>Note that this `email` must match the one that was used to sign up for an account via our website.</u> Please wait for this confirmation before editing the metadata or attempting to publish your result.


Note that submissions are hidden by default and that we provide the ability for users to "publish" or un-hide their results using the website. Submissions may also be hidden after they are published if desired.

## Submission Policy

The submission page of our website should only be used for reporting final results for comparison with other methods on our website. 

For the Boreas dataset benchmark, our expectation is that users will iterate their algorithms on a validation set based on the provided training data and only submit their best results on the test set once. Users are prohibited from tuning their algorithms to achieve better results on the test set. If users desire to compare the performance of multiple versions of the same algorithm, the provided training data should be used for this purpose.

Incorrect submission formatting resulting in erroneous or failed benchmark results does not count as a proper submission.

## Anonymous Submissions

Similar to the KITTI dataset, we want to avoid our leaderboard being cluttered by anonymous submissions. We may periodically hide or delete anonymous submissions that have not been associated with a peer-reviewed publication after a period of 6 months.
