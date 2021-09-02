from scipy.spatial.transform import Rotation as R


class LidarScan:
    def __init__(self, ros_ts, gps_ts, position, heading, points):
        self.ros_ts = ros_ts
        self.gps_ts = gps_ts
        self.position = position
        self.heading = heading  # roll, pitch, heading(yaw)
        self.points = points

    def get_C_v_enu(self):
        return R.from_euler('xyz', self.heading)