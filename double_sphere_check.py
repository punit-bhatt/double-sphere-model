import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from double_sphere_model import DoubleSphereCamera
from generate_mask import get_image_mask

def get_projection_error(mask, camera):
    """Gets the projection error at different pixel locations. This is done by
    projecting the pixel location to 3D coordinates (this would just be a ray).
    The normalized 3D coordinates are then reprojected to the image plane. The
    new pixel location's now compared with the original one and this separation
    gives the projection error.

    Args:
        mask : Binary image mask.
        camera : Camera model initialized with appropriate parameters.

    Returns:
        Projection errors across the image.
    """

    points_2d = torch.from_numpy(np.argwhere(mask > 0.))
    points_projected_3d = camera.project_2d_to_3d(points_2d)

    norm_projected_3d = points_projected_3d / \
        torch.linalg.norm(points_projected_3d, dim=1, keepdims=True)

    points_projected_2d = camera.project_3d_to_2d(norm_projected_3d)

    heatmap_xy = np.zeros_like(mask)

    rmse = torch.sqrt(torch.sum((points_2d - points_projected_2d) ** 2, dim=1))
    heatmap_xy[points_2d[:, 0], points_2d[:, 1]] = rmse

    plt.imshow(heatmap_xy, interpolation = 'nearest')
    plt.title('Double Sphere Model - Projection Error')
    plt.colorbar()
    plt.show()

    print(f'\n{torch.min(rmse) = }')
    print(f'{torch.max(rmse) = }')
    print(f'{torch.mean(rmse) = }')
    print(f'{torch.std(rmse) = }')

    return heatmap_xy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image-path',
                        help="Defines the image path.",
                        default=None)
    parser.add_argument('-m',
                        '--mask-path',
                        help="Defines the binary image mask path.",
                        default=None)
    parser.add_argument('-p',
                        '--parameters',
                        help="Defines the camera model parameters.",
                        nargs='+',
                        type=float,
                        default=[-0.17023409,
                                 0.59679147,
                                 156.96507623,
                                 157.72873153,
                                 343,
                                 343])
    args = parser.parse_args()
    image_path = args.image_path
    mask_path = args.mask_path
    parameters = args.parameters

    assert mask_path is not None or image_path is not None

    print(f'{image_path = }')
    print(f'{mask_path = }')
    print(f'{parameters = }')

    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = get_image_mask(image_path)

    camera = DoubleSphereCamera(parameters)

    get_projection_error(mask, camera)
