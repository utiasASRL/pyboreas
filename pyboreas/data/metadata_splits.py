"""
Common metadata splits for convenience
Each split is a list of lists, where each inner list contains the sequence ID and optionally start and end timestamps (in microseconds) for the evaluation interval.
The following splits are provided for convenience:
- route splits: splits of sequences based on the route they were collected on
- tag splits: splits of sequences based on metadata tags (e.g. weather, environment, etc.)

This file also includes temporal splits for sensor upgrades (in unix time) that can be used to split sequences based on when they were collected relative to upgrades.

Feel free to add additional splits as needed through submitting a PR.
"""

# Temporal splits (sensor upgrade times in Unix epoch seconds)
radar_resolution_upgrade_time = 1632182400  # before: resolution = 0.0596, after: resolution = 0.04381
radar_chirp_type_upgrade_time = 1733248400  # before: no chirp type data, after: chirp type data included

# List: [ID]

## ROUTE SPLITS
urban_split = [
    ["boreas-2025-08-06-06-33"], # urban
    ["boreas-2025-08-06-07-05"], # urban
    ["boreas-2025-08-06-07-41"], # urban
    ["boreas-2025-08-06-08-35"], # urban
    ["boreas-2025-08-06-10-48"], # urban
    ["boreas-2025-08-06-11-32"], # urban
    ["boreas-2025-08-06-12-20"], # urban
]

## TAG SPLITS
aeva_split = [  # FMCW Lidar tag on download page
    ["boreas-2024-12-03-12-54"], # suburbs
    ["boreas-2024-12-05-14-25"], # suburbs
    ["boreas-2025-01-08-10-59"], # suburbs
    ["boreas-2025-01-08-11-22"], # suburbs
    ["boreas-2025-01-08-12-28"], # suburbs
    ["boreas-2024-12-03-13-13"], # regional north
    ["boreas-2024-12-03-13-34"], # regional south
    ["boreas-2024-12-10-12-07"], # regional north
    ["boreas-2024-12-10-12-24"], # regional south
    ["boreas-2024-12-10-12-38"], # regional north
    ["boreas-2024-12-10-12-56"], # regional south
    ["boreas-2024-12-04-14-28"], # tunnel east
    ["boreas-2024-12-04-14-34"], # tunnel west
    ["boreas-2024-12-04-14-38"], # tunnel east
    ["boreas-2024-12-04-14-44"], # tunnel west
    ["boreas-2024-12-04-14-50"], # tunnel east
    ["boreas-2024-12-04-14-59"], # tunnel west
    ["boreas-2024-12-04-15-04"], # tunnel east
    ["boreas-2024-12-04-15-10"], # tunnel west
    ["boreas-2024-12-04-15-19"], # tunnel east
    ["boreas-2024-12-04-15-24"], # tunnel west
    ["boreas-2024-12-04-11-45"], # skyway
    ["boreas-2024-12-04-11-56"], # skyway
    ["boreas-2024-12-04-12-08"], # skyway
    ["boreas-2024-12-04-12-19"], # skyway
    ["boreas-2024-12-04-12-34"], # skyway
    ["boreas-2024-12-05-14-12"], # industrial
    ["boreas-2024-12-23-16-27"], # industrial
    ["boreas-2024-12-23-16-44"], # industrial
    ["boreas-2024-12-23-17-01"], # industrial
    ["boreas-2024-12-23-17-18"], # industrial
]
