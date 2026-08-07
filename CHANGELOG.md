# Changelog

All notable changes to this project will be documented in this file.

## [2026-08-07]

### Changed
- Updated camera intrinsic and camera-lidar extrinsic calibration for Boreas Road Trip sequences.
- Updated calibration procedure:
    - Collected new calibration data using a more rigid checkerboard to better satisfy the planar calibration-target assumption.
    - Used the [ROS1 camera_calibration](https://wiki.ros.org/camera_calibration) package during data collection to assess image coverage across the camera x-y plane, target skew, and scale.
        - Prioritized close-up, skewed views to maximize the information and constraints provided by checkerboard corners in each image.
    - Used MATLAB [Camera Calibrator](https://www.mathworks.com/help/vision/ref/cameracalibrator-app.html) and [Lidar Camera Calibrator](https://www.mathworks.com/help/lidar/ref/lidarcameracalibrator-app.html) packages to update the intrinsic and extrinsic calibrations.
- Images in BRT sequences have been be undistorted using the updated camera intrinsics.

## [2025-07-09]

### Added
- Added the `boreas-objects-v1` sequence to the dataset website's download page.
