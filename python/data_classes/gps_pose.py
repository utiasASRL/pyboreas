from scipy.spatial.transform import Rotation as R


class GPSPose:
    def __init__(self, gps_ts, position, heading):
        self.gps_ts = gps_ts
        self.position = position
        self.heading = heading  # roll, pitch, heading(yaw)

    def get_C_vo(self):
        return R.from_euler('xyz', self.heading)
