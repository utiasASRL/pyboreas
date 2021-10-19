# A Guide to the Boreas Dataset

## Introduction

### Purpose

This dataset and the associated benchmarks are intended to support odometry and metric localization for lidar, radar, and vision. In the future, we hope to also be able to provide 3D and 2D object labels. This dataset features repeated traversals over a long period and multiple weather conditions. These changing conditions may be used to benchmark long-term localization capabilities of different sensors or the robustness of various sensor types to adverse weather conditions.

### Sensors

- 128-beam Velodyne Alpha-Prime 3D lidar
- FLIR Blackfly S (5 MP) monocular camera
- Navtech 360$^\circ$ radar
- Applanix POSLV GNSS

### Data Collection

Data was collected during repeated traversals of several routes in Toronto, Canada, across several seasons and multiple weather conditions. Two of these routes are shown below:

| ![sat_dt.png](figs/glen_shields.png) | ![glen_shields.png](figs/st_george.png) |
| --- | --- |
| [Glen Shields](figs/st_george.html) | Glen Shields |