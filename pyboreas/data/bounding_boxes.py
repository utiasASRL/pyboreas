import numpy as np


class BoundingBox2D:
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

        # Construct array to extract points from extent
        dims_multiplier = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                                    [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]])

        # TODO: need orientation here
        points = []
        for i in range(dims_multiplier.shape[0]):
            points.append(position + extent * dims_multiplier[i].reshape(3, 1) / 2)
        self.points = np.concatenate(points, axis=-1)  # Top 4 points ccw, then bottom 4 points, ccw. Dims=(3, #pts)

    def render_bbox_2d(self, ax, color="r"):
        """Render the bbox into a top-down 2d view

        Args:
            ax: the axis to render the bbox onto
        """
        prev_pt = self.points[:, 3]
        for i in range(4):  # Just draw top 4 points of bbox
            ax.plot([prev_pt[0], self.points[0, i]], [prev_pt[1], self.points[1, i]], color=color)
            prev_pt = self.points[:, i]

    def get_points_perspective(self, view: np.ndarray, normalize: bool) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
         all zeros) and normalize=False

        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = self.points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((self.points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def get_raw_data(self):
        return self.pos, self.rot, self.extent
