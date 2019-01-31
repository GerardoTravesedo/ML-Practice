import cv2


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


def show_image_with_bboxes(image, bboxes):
    """
    :param
        image: pixels representing the image (BGR).
        bboxes: list([x, y, w, h)] representing the top-left corner of the box along with the
        width and height
    """
    color = (0, 255, 0)
    thickness = 2

    for bbox in bboxes:
        top_left_pixel = (bbox["x"], bbox["y"])
        bottom_right_pixel = (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
        cv2.rectangle(image, top_left_pixel, bottom_right_pixel, color, thickness)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

