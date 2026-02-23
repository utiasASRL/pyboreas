from pathlib import Path
import os.path as osp
import numpy as np

from pyboreas.utils.utils import micro_to_sec, get_closest_index

# File adapted from pycanoe repository developed by Mia Thomas.

class AuxSensor:
    """Class for auxillary sensors.

    Sensor data stored in csv (instead of individual timestamped files as in regular Sensors). Missing some fields compared to main Sensors (e.g., pose, body rate, velocity).
    """

    def __init__(self, csv_path, timestamp_micro):
        self.csv_path = csv_path

        self.sensType = None
        self.seq_root = None
        self.sensor_root = None
        self.seqID = None

        self.frame = np.int64(timestamp_micro)
        self.timestamp_micro = np.int64(timestamp_micro)
        self.timestamp = micro_to_sec(timestamp_micro)

        # Parse directories
        p = Path(csv_path)
        if len(p.parts) >= 2:
            self.sensType = p.parts[-2]
            self.seq_root = str(Path(*p.parts[:-2]))
            self.sensor_root = osp.join(self.seq_root, self.sensType)
        if len(p.parts) >= 3:
            self.seqID = p.parts[-3]

        # NOTE: Aux Sensors have no pose, velocity, bodyrate


# TODO: grab all within range


class AuxCSV:
    """Singleton class for managing the CSVs that hold the auxillary sensor data.

    Assumes first row is header and first column is timestamp.
    Default timestamp is converted to seconds by the specified timestamp_multiplier (e.g., 1 for seconds, 1e6 for microseconds).
    """

    _instances = {}

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use AuxCSV.get_instance(csv_path) to load class.")

    @classmethod
    def get_instance(cls, csv_path, timestamp_multiplier):
        """Manage & return instances of csv (one instance per unique csv file)"""
        # Normalize for comparison
        norm_path = osp.realpath(osp.expanduser(csv_path))

        # New instance for each unique csv
        if norm_path not in cls._instances:
            cls._instances[norm_path] = cls._create(norm_path, timestamp_multiplier)

        return cls._instances[norm_path]

    @classmethod
    def _create(cls, norm_path, timestamp_multiplier):
        """Internal: Create new instance, bypassing __init__"""
        instance = object.__new__(cls)
        instance._initialize(norm_path, timestamp_multiplier)
        return instance

    def _initialize(self, norm_path, timestamp_multiplier):
        """Internal: Initialize new instance, bypassing __init__"""
        self.csv_path = norm_path
        self.data = None
        self.timestamp_multiplier = timestamp_multiplier
        self.timestamps_micro = None
        # Dictionary of timestamp -> index for O(1) retrieval
        self.timestamps_micro_index = None

        if not osp.exists(self.csv_path):
            print(f"WARNING: Could not find file {self.csv_path}")

    def load_csv(self):
        """Load csv data, populate timestamps (micro) and timestamp micro lookup dict"""
        if self.data is None:
            data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)
            self.data = data
            conversion_factor = self.timestamp_multiplier * 1e6  # Convert to microseconds
            self.timestamps_micro = (data[:, 0] * conversion_factor).astype(np.int64)
            self.timestamps_micro_index = {
                ts: idx for idx, ts in enumerate(self.timestamps_micro)
            }

    def get_at_timestamp_micro(self, ts_micro: int):
        self.load_csv()

        if ts_micro not in self.timestamps_micro_index:
            raise ValueError(f"Timestamp {ts_micro} not found in CSV.")

        idx = self.timestamps_micro_index[ts_micro]
        return self.data[idx]

    def get_closest(self, ts_micro: int):
        self.load_csv()

        # Exact
        if ts_micro in self.timestamps_micro_index:
            idx = self.timestamps_micro_index[ts_micro]
        # Closest
        else:
            idx = get_closest_index(ts_micro, self.timestamps_micro)
            if (
                ts_micro < self.timestamps_micro[0]
                or ts_micro > self.timestamps_micro[-1]
            ):
                print(
                    f"WARNING: Timestamp {ts_micro} is out of CSV range "
                    f"{[self.timestamps_micro[0], self.timestamps_micro[-1]]}."
                )
        return self.data[idx]

    def get_all_timestamps_micro(self):
        self.load_csv()
        return self.timestamps_micro

class IMU(AuxSensor):
    """IMU data

    Attributes:
        body_angvel (np.array): body angular velocity (rad/s)
        body_acc (np.array): body linear acceleration, gravity not removed (m/s^2)
    """

    def __init__(self, csv_path, timestamp_micro):
        AuxSensor.__init__(self, csv_path, timestamp_micro)
        self.body_angvel = None
        self.body_acc = None
        self.timestamp_multiplier = None  # To be set by child class, depending on timestamp units in csv (e.g., 1 for seconds, 1e6 for microseconds)

    # Can change this within individual instances if the column layout differs
    def _parse_line(self, line):
        """Parse a CSV row into IMU fields.

        Subclasses may override this if their column layout differs.
        """
        _, wx, wy, wz, ax, ay, az = line
        body_angvel = np.array([wx, wy, wz])
        body_acc = np.array([ax, ay, az])
        return body_angvel, body_acc

    def load_data(self):
        csv = AuxCSV.get_instance(self.csv_path, timestamp_multiplier=self.timestamp_multiplier)
        line = csv.get_at_timestamp_micro(self.timestamp_micro)

        self.body_angvel, self.body_acc = self._parse_line(line)

    def unload_data(self):
        self.body_angvel = None
        self.body_acc = None

class DMU(IMU):
    """DMU data, identical to IMU."""
    def __init__(self, csv_path, timestamp_micro):
        super().__init__(csv_path, timestamp_micro)
        self.timestamp_multiplier = 1e-9  # DMU uses nanoseconds

class AevaIMU(IMU):
    """Aeva IMU data, identical to IMU."""
    def __init__(self, csv_path, timestamp_micro):
        super().__init__(csv_path, timestamp_micro)
        self.timestamp_multiplier = 1e-6 # Aeva IMU uses microseconds

class Encoder(AuxSensor):
    """Encoder data

    Attributes:
        pulse_count (int): cumulative pulse count
    """

    def __init__(self, csv_path, timestamp_micro):
        AuxSensor.__init__(self, csv_path, timestamp_micro)
        self.pulse_count = None

    def load_data(self):
        csv = AuxCSV.get_instance(self.csv_path)
        line = csv.get_at_timestamp_micro(self.timestamp_micro)

        _, pulse_count = line
        self.pulse_count = int(pulse_count)

    def unload_data(self):
        self.pulse_count = None