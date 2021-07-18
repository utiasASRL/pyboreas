import numpy as np

class BBox:
    def __init__(self, position, rotation, extent, label=None):
        """Checks dimensional consistency of inputs and constructs points array

        Args:
            position: (x,y,z) position of bbox centroid
            rotation: rotation matrix for bbox orientation
            extent: (width, length, height) of the bbox
            label: optional string for bbox label
        """
        assert position.shape == (3, 1)
        assert rotation.shape == (3, 3)
        assert extent.shape == (3, 1)
        self.pos = position
        self.rot = rotation
        self.extent = extent
        self.label = label

        # Construct points array
        dims_multiplier = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                                    [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        points = []
        for i in range(dims_multiplier.shape[0]):
            points.append(position + extent * dims_multiplier[i].reshape(3, 1) / 2)
        self.points = np.concatenate(points, axis=-1)  # Top 4 points ccw, then bottom 4 points, ccw

    def render_bbox_2d(self, ax, color="r"):
        """Render the bbox into a top-down 2d view

        Args:
            ax: the axis to render the bbox onto
        """
        prev_pt = self.points[:, 3]
        for i in range(4):  # Just draw top 4 points of bbox
            ax.plot([prev_pt[0], self.points[0, i]], [prev_pt[1], self.points[1, i]], color=color)
            prev_pt = self.points[:, i]

    def get_raw_data(self):
        return self.pos, self.rot, self.extent



