Submissions are zip files which contain the result txt files along with a metadata.yaml file.

Note that train/test splits are contained in `pyboreas.data.splits.py`

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
- `2d` if True, evaluate using SE(2), otherwise submissions are evaluated in SE(3).
- `author, papertitle, paperurl, venue` are optional tags which can be left blank for an anonymous submission and may be updated via the website later.

For the localization benchmark, an additional metadata tag is required: `reference` which can be one of: `lidar, radar, camera`. This is used to determine which sensor is being used as a reference for ground truth poses.

`author` is a comma-separated list of author names.

Note that submissions are hidden by default and that we provide the ability for users to "publish" or un-hide their results using the website.

Note that we provide a Python file to check the format of the submission zip file: `pyboreas.eval.submission_checker.py`. Please use this script to check your submission before uploading to the website. Your submission may fail silently if this has not been done.

See the other readme files for the format of the submission txt files.

After uploading your submission, it will take several minutes before your results will apper in the "view / edit" section of the submission page.

Detailed results will be sent to the `email` provided in the yaml file.