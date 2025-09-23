# Evaluation

Submissions are zip files which contain the result txt files along with a metadata.yaml file.

Note that train/test splits are contained in `pyboreas.data.splits.py`
Users are free to create their own train/validation splits from the available training data.

The name of the zip file must end in the desired benchmark which can be one of: `[odometry, localization, detection]`. Example: `boreas-odometry.zip`. The benchmark specified in the file name must match the benchmark specified in the yaml file.

metadata.yaml uses the following format:

```YAML
benchmark: odometry
methodname: RoBoDoMeTrY
email: youremail@gmail.com
2d: False
author: First Last, First1 Last1
papertitle: Our Wondrous Algorithm
paperurl: https://www.website/paperURL
venue: Conference or Journal Name
year: 2077
runtimeseconds: 0.1
computer: Intel Xeon CPU E3-1505M v6 @ 3.00GHz
```

- `benchmark` can be one of: `[odometry, localization, detection]`.
- `methodname` is a short nickname for your method
- `2d` if True, evaluate using SE(2) or BEV for object detections, otherwise submissions are evaluated in SE(3) or 3D for object detections.
- `author, papertitle, paperurl, venue` are optional tags which can be left blank for an anonymous submission and may be updated via the website later.

For the localization benchmark, two additional metadata tags are required: 
`ref_sensor` which can be one of: `lidar, radar, camera`. This is used to determine which sensor is being used as a reference for ground truth poses.
`test_sensor` which can be one of: `lidar, radar, camera`. This is used to determine which sensor acts as the "test" sensor.

`author` is a comma-separated list of author names.

Note that submissions are hidden by default and that we provide the ability for users to "publish" or un-hide their results using the website. Submissions may also be hidden after they are published if desired.

Note that we provide a Python file to check the format of the submission zip file: `pyboreas.eval.submission_checker.py`. Please use this script to check your submission before uploading to the website. Your submission may fail silently if this has not been done.

See the other readme files for the format of the submission txt files.

After uploading your submission, it will take several minutes before your results will apper in the "view / edit" section of the submission page.

Detailed results will be sent to the `email` provided in the yaml file. Note that this `email` must match the one that was used to sign up for an account via our website. You should have received an Amazon Web Services confirmation link after confirming your account creation. Is this was not the case, please email us.

## Submission Policy

The submission page of our website should only be used for reporting final results for comparison with other methods on our website. Our expectation is that users will iterate their algorithms on a validation set based on the provided training data and only submit their best results on the test set once. Users are prohibited from tuning their algorithms to achieve better results on the test set. If users desire to compare the performance of multiple versions of the same algorithm, the provided training data should be used for this purpose.

Incorrect submission formatting resulting in erroneous or failed benchmark results does not count as a proper submission.

## Anonymous Submissions

Similar to the KITTI dataset, we want to avoid our leaderboard being cluttered by anonymous submissions. We may periodically hide or delete anonymous submissions that have not been associated with a peer-reviewed publication after a period of 6 months.
