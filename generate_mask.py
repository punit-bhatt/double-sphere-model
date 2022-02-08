import argparse
import cv2

def get_image_mask(image_path, output_path=None):
    """Generates a binary mask from the given image.

    Args:
        image_path : Fisheye image file path.
        output_path : Binary mask output file path.

    Returns:
        Binary mask.
    """

    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0.] = 255

    if output_path is None:
        output_path = '.'.join([image_path.split('.')[0] + '-mask',
                            image_path.split('.')[-1]])

    cv2.imwrite(output_path, mask)

    return mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image-path',
                        help="Defines the image path.",
                        required=True)
    parser.add_argument('-o',
                        '--output-path',
                        help="Defines the output path for the generated mask.",
                        default=None)

    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path

    get_image_mask(image_path, output_path)