import torch

class DoubleSphereCamera:

    def __init__(self, parameters):
        """Initializes a double sphere camera model instance.

        Args:
            parameters : List of parameters ordered as follows -
                chi, alpha, fx, fy, cx, cy
        """

        self.chi, self.alpha, self.fx, self.fy, self.cx, self.cy = parameters


    def project_3d_to_2d(self, points):
        """Projects a tensor of 3d points to their 2d counterparts on the image
        plane.

        Args:
            points : A (N, 3) tensor containing all the 3d points.

        Returns:
            2d points.
        """

        assert len(points.shape) == 2
        assert points.shape[1] == 3

        x, y, z = points.T

        d1 = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        d2 = torch.sqrt(x ** 2 + y ** 2 + (self.chi * d1 + z) ** 2)

        denominator = self.alpha * d2 + (1 - self.alpha) * (self.chi * d1 + z)

        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))

    def project_2d_to_3d(self, points):
        """Unprojects a tensor of pixel locations (2d points) to the 3d camera
        coordinate system for an arbitrary z.

        Args:
            points : A (N, 2) tensor containing the pixel locations / 2d points.

        Returns:
            3d points.
        """

        assert len(points.shape) == 2
        assert points.shape[1] == 2

        u, v = points.T

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r_2 = mx ** 2 + my ** 2
        mz = (1 - (self.alpha ** 2) * r_2) / \
            (self.alpha * torch.sqrt(1 - (2 * self.alpha - 1) * r_2) + \
                (1 - self.alpha))

        coefficient = (mz * self.chi + \
            torch.sqrt(mz ** 2 + (1 - self.chi ** 2) * r_2)) / \
                (mz ** 2 + r_2)

        x = coefficient * mx
        y = coefficient * my
        z = coefficient * mz - self.chi

        return torch.hstack((x.reshape(-1, 1),
                             y.reshape(-1, 1),
                             z.reshape(-1, 1)))
