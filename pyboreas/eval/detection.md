# 3D Object Benchmarking

## Ground Truth Format

Label formats:

Note that the name of each lidar file is a UNIX timestamp in microseconds and corresponds to the timestamp of the associated lidar frame that was labelled.

The table below describes each column of the label files:

| Values | Name       | Description                                                          |
| ------ | ---------- | -------------------------------------------------------------------- |
| 1      | uuid       | Unique identifier for the object track, consistent across frames     |
| 1      | type       | Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', 'Misc' |
| 3      | dimensions | 3D object dimensions: length, width, height (in meters)              |
| 3      | location   | 3D object location x,y,z in the lidar frame (in meters)              |
| 1      | rotation_z | Rotation around Z-axis in lidar frame [-pi..pi]                      |
| 1      | numPoints  | The number of lidar points associated with the bounding box.         |

Objects are labelled +/- 75m from the lidar sensor frame, in a square area centered on the vehicle.

Pedestrian class: any person (adult, child, or baby) in any position. Exclusions: pedestrians in buildings, cars, balconies, or an image of a person. Includes: skateboarders, people riding scooters.

Cars includes: coupes, sedans, SUV, Vans, pick-up trucks, ambulances.

Cyclist include: bicyclists, motorcyclists. Exclusion: parked bicycles, parked motorcycles.

Misc includes: bus, industrial truck, streetcar, train

Note that we provide two similar sources of ground truth labels: `labels_detection` is a subset of `labels_tracking`. For `labels_detection`, we have removed bounding box labels with less than 25 lidar points. We have done the same for the ground truth detection labels. Please remove objects associated with less than 25 lidar points from your submission as these will be counted as false positives.

## Prediction Format

For the detection benchmark, you must submit one label file for each file in the test set. Defined in `detection_test_times.txt`.

Each row of a label file will correspond to a predicted bounding box label with the following format.
The table below describes each column of the label files:
Values in the label files are separated by spaces.

| Values | Name       | Description                                                                         |
| ------ | ---------- | ----------------------------------------------------------------------------------- |
| 1      | uuid       | Set to a default value, like -1, this value is ignored for the detection benchmark. |
| 1      | type       | Describes the type of object: 'Car', 'Pedestrian', 'Cyclist'                        |
| 3      | dimensions | 3D object dimensions: length, width, height (in meters)                             |
| 3      | location   | 3D object location x,y,z in the lidar frame (in meters) in lidar frame              |
| 1      | rotation_z | Rotation around Z-axis in lidar frame [-pi..pi]                                     |
| 1      | numPoints  | Set to a default value, like -1, this value is ignored for the detection benchmark. |
| 1      | score      | Confidence in detection, needed for p/r curves                                      |

Note: miscellaneous vehicle types (Misc) will be ignored by the benchmark. Avoid labelling these types of vehicles as 'Car' as that will be counted as a false positive by the benchmark.
