import cv2
import random


def image_to_pixels(image):
    """
    Given the path to an image, it returns the numpy object with the pixels
    For example: Image 2007_000027.jpg returns a numpy object with shape (500, 486, 3)

    :param
        image: path to an image
    """
    return cv2.imread(image)


def resize_image(image, height, width):
    """
    Given an image, it first converts it into shape (32 x 32 x 3) and then
    it resizes it to (height x width x 3)

    :param
        image: raw image (pixels).
        height: new height for the image
        width: new width for the image
    """
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def show_image_from_pixels(image):
    """
    :param
        image: pixels representing the image (BGR).
    """
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_image_with_bboxes(image, image_name, roi_bboxes, gt_bboxes, output_folder):
    """
    Draws foreground rois and ground truth box on the original image and writes the image to the
    output folder

    :param
        image: pixels representing the image (BGR).
        image_name: name that it will give to output image it generates
        roi_bboxes: list([x, y, w, h)] representing the top-left corner of the box along with the
        width and height
        gt_bboxes: list([x, y, w, h)] representing the top-left corner of the box along with the
        width and height
        output_folder: folder where all the generated images will be written
    """
    def draw_bbox(box, color):
        top_left_pixel = (box[0], box[1])
        bottom_right_pixel = (box[0] + box[2], box[1] + box[3])
        cv2.rectangle(image, top_left_pixel, bottom_right_pixel, color, thickness)

    gt_bbox_color = (0, 255, 0)
    thickness = 2

    for bbox in roi_bboxes:
        red = random.choice(range(256))
        green = random.choice(range(256))
        blue = random.choice(range(256))
        draw_bbox(bbox, (red, green, blue))

    for bbox in gt_bboxes:
        draw_bbox(bbox, gt_bbox_color)

    cv2.imwrite(output_folder + image_name, image)


