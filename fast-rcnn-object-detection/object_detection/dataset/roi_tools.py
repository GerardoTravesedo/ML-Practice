import selectivesearch
import math
import numpy as np

NUMBER_CLASSES = 21


# USED THIS METHOD ONLY WHEN TRAINING
def find_rois_complete(image_pixels, gt_boxes, min_rois_foreground, max_rois_background):
    """
    Generates a minimum number of foreground rois and a maximum number of background ones

    :param
        image_pixels: pixels from the image
        gt_boxes: Ground truth boxes from the image
        min_rois_foreground: minimum number of foreground rois to find
        max_rois_background: maximum number of background rois to find
    """
    # Map of final foreground rois (needs to be at least min_rois_foreground)
    # We use a map because we want a unique set of positive rois
    rois_foreground = {}
    # List of final background rois (needs to be at most max_rois_background)
    rois_background = []
    # Initial scale for selective search. It will be reduces to find more foreground rois
    init_scale = 500

    # Iterate until we have the required number of foreground rois
    while len(rois_foreground) < min_rois_foreground and init_scale > 100:
        # Finding rois for current scale
        rois = find_rois_selective_search(image_pixels, scale=init_scale)
        # For each roi, we find if it is foreground or background
        for roi in rois:
            roi_info = find_roi_labels(roi, gt_boxes)
            if len(rois_background) < max_rois_background and roi_info["class"][0] == 1:
                rois_background.append(roi_info)
            elif roi_info["class"][0] == 0:
                # Keeping only unique positive rois (there could be duplicates coming
                # from different calls to find_rois with different scales)
                key = str(roi[0]) + str(roi[1]) + str(roi[2]) + str(roi[3])
                rois_foreground[key] = roi_info
        # Reducing scale for next iteration of selective search
        init_scale = init_scale - 100

    # If selective search couldn't find any positive rois even trying multiple parameters, we
    # generate our own positive rois by moving the ground truth box slightly
    if len(rois_foreground) == 0:
        return find_rois_from_ground_truth_boxes(gt_boxes, image_pixels.shape), rois_background
    else:
        return np.array(rois_foreground.values()), np.array(rois_background)


def find_rois_selective_search(image_pixels, scale=200, sigma=0.9, min_size=10):
    """
    Uses the selective search library to find rois


    :param
        image_path: path to an image
        image_pixels: pixels from the image
    """
    # Higher scale means higher preference for larger components (k / |C|, where |C| is the
    # number of pixels in the component and k is the scale; for a large k, it would be difficult
    # for a small component to be have a separation boundary with the neighboring components since
    # the division is large). Smaller components are allowed when there is a sufficiently large
    # difference between neighboring components (the higher k / |C|, the higher the difference
    # between neighboring components has to be)
    img_lbl, regions = \
        selectivesearch.selective_search(image_pixels, scale=scale, sigma=sigma, min_size=min_size)

    unique_rois = {}

    # Deduplicating rois
    for region in regions:
        # rect format: [x, y, w, h]
        rect = region["rect"]
        key = str(rect[0]) + str(rect[1]) + str(rect[2]) + str(rect[3])
        if key not in unique_rois:
            # From [x, y, w, h] to {x, y, w, h}
            unique_rois[key] = rect

    return np.array(unique_rois.values())


def find_rois_from_ground_truth_boxes(gt_boxes, image_shape):
    """
    Finds foreground rois from the ground truth boxes. It creates 4 foreground rois for each 
    ground truth box by moving the box a little bit to the right, left, up and down

    :param
        gt_boxes: Ground truth boxes from the image
        image_shape: shape of the image that contains the boxes
    """
    image_height_pixels = image_shape[0]
    image_width_pixels = image_shape[1]
    foreground_rois = []

    for gt_box in gt_boxes:
        gt_box = gt_box["bbox"]

        max_x = image_width_pixels - gt_box[2]
        max_y = image_height_pixels - gt_box[3]

        # Move gt box to the right
        new_x = gt_box[0] + (gt_box[2] / 4) - 1
        if new_x < max_x:
            foreground_rois.append([new_x, gt_box[1], gt_box[2], gt_box[3]])
        # Move gt box to the left
        new_x = gt_box[0] - (gt_box[2] / 4) + 1
        if new_x > 0:
            foreground_rois.append([new_x, gt_box[1], gt_box[2], gt_box[3]])
        # Move gt_box up
        new_y = gt_box[1] - (gt_box[3] / 4) + 1
        if new_y > 0:
            foreground_rois.append([gt_box[0], new_y, gt_box[2], gt_box[3]])
        # Move gt_box down
        new_y = gt_box[1] + (gt_box[3] / 4) - 1
        if new_y < max_y:
            foreground_rois.append([gt_box[0], new_y, gt_box[2], gt_box[3]])

    return np.array(foreground_rois)


def calculate_iou(gt_bbox, roi_bbox):
    """
    Calculates intersection over union between the ground truth bbox and a particular roi bbox

    :param
        gt_bbox: ground truth bbox
        roi_bbox: region of interest bbox
    """
    # Calculating corners of intersection box
    # Top left corner
    intersect_top_left_x = max(gt_bbox[0], roi_bbox[0])
    intersect_top_left_y = max(gt_bbox[1], roi_bbox[1])
    # Bottom right corner
    intersect_bottom_right_x = \
        min(gt_bbox[0] + gt_bbox[2] - 1, roi_bbox[0] + roi_bbox[2] - 1)
    intersect_bottom_right_y = \
        min(gt_bbox[1] + gt_bbox[3] - 1, roi_bbox[1] + roi_bbox[3] - 1)

    # We add +1 because the two boxes could be overlapping on one line of pixels (one edge), and
    # that shouldn't count as 0
    area_intersection = max(0, intersect_bottom_right_x - intersect_top_left_x + 1) * \
        max(0, intersect_bottom_right_y - intersect_top_left_y + 1)

    area_gt_bbox = gt_bbox[2] * gt_bbox[3]
    area_roi_bbox = roi_bbox[2] * roi_bbox[3]

    union_area = area_gt_bbox + area_roi_bbox - area_intersection

    return area_intersection / float(union_area)


def find_roi_labels(roi_bbox, gt_objects):
    """
    Generates labels for a given roi. The labels are composed of a class and a bbox regression
    target.

    The class is found by calculating the IoU with all the ground truth boxes and keeping the
    class of the one with highest value

    The regression targets are found using the following formulas:

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = log(Gw / Pw)
    th = log(Gh / Ph)

    :param
        roi_bbox: region of interest bbox
        gt_objects: all the objects in the image (contains class and bbox)
    """
    max_iou = 0.5
    roi_class = None
    roi_bbox_target = np.zeros(1)

    # Finding the gt object with the highest IoU with the roi
    for gt_object in gt_objects:
        iou = calculate_iou(gt_object["bbox"], roi_bbox)

        if iou >= max_iou:
            max_iou = iou
            roi_class = gt_object["class"]
            roi_bbox_target = gt_object["bbox"]

    # If roi_bbox_target only has zeros, any returns false
    if roi_class and roi_bbox_target.any():
        # Calculating regression targets according to formulas on paper
        tx = (roi_bbox_target[0] - roi_bbox[0]) / float(roi_bbox[2])
        ty = (roi_bbox_target[1] - roi_bbox[1]) / float(roi_bbox[3])
        tw = math.log(roi_bbox_target[2] / float(roi_bbox[2]))
        th = math.log(roi_bbox_target[3] / float(roi_bbox[3]))
        # [tx, ty, tw, th]
        regression_targets = [tx, ty, tw, th]
        return {"bbox": np.array(roi_bbox),
                "class": class_string_to_index(roi_class),
                "reg_target": np.array(regression_targets)}
    else:
        # If roi doesn't have IoU > 0.5 with any gt object, then it is background and it doesn't
        # have regression targets
        return {"bbox": np.array(roi_bbox),
                "class": class_string_to_index("background"),
                "reg_target": np.zeros(4)}


def class_string_to_index(class_string):
    """
    Converts a class in string format into an array with all values to 0 but a 1 for the index
    of the right class

    :param
        class_string: string representing the class name
    """
    switcher = {
        "background": 0, "person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5, "horse": 6,
        "sheep": 7, "aeroplane": 8, "bicycle": 9, "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
        "train": 14, "bottle": 15, "chair": 16, "dining table": 17, "potted plant": 18,
        "sofa": 19, "tv/monitor": 20
    }

    class_index = switcher.get(class_string, -1)

    if class_index == -1:
        raise Exception("Invalid class " + class_string)

    classes = np.zeros(NUMBER_CLASSES)
    classes[class_index] = 1

    return classes
