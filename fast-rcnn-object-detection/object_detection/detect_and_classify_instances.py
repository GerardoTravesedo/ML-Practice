import tensorflow as tf
import numpy as np

r = tf.placeholder(tf.float32, shape=(None, 4))
r_scores = tf.placeholder(tf.float32, shape=(None, 3))
r_reg = tf.placeholder(tf.float32, shape=(None, 2, 4))


NUMBER_CLASSES = 3

def detect_and_classify(
        rois, rois_class_scores, rois_reg_targets, max_output_size, score_threshold):
    """

    :param rois: rois found for the current images. Shape = (# rois, # bbox fields = 4)
    :param rois_class_scores: Class scores for the different rois.
        Shape = (# rois, # classes)
    :param rois_reg_targets: Regression targets for each roi for each class.
        Shape = (# rois, # classes - background, # reg fields = 4)
    :param max_output_size: maximum number of boxes to be selected by non max suppression
    :param score_threshold: hreshold for deciding when to remove boxes based on score

    :return: All instances of objects detected along with their classes
        Shape = (# objects detected, # detection fields = 5) where detection fields contain:
            - 4 fields for the bbox (x, y, w, h)
            - 1 field for the class
    """
    # Removing background class since we don't want to find objects for it
    no_background_class = rois_class_scores[:, 1:]
    roi_class_scores_shape = tf.shape(no_background_class)
    # Reshaping class scores. Example: [[0.3, 0.6], [0.6, 0.3]] -> [[[0.3], [0.6]], [[[0.6], [0.3]]]
    # In the example above we have two rois, two classes (after removing background)
    # We do this to be able to concat it to the reg target tensor
    reshaped_class_scores = tf.reshape(
        no_background_class, shape=[roi_class_scores_shape[0], roi_class_scores_shape[1], 1])
    # Concat reg targets (we have one group per roi per class) with the corresponding class scores
    # There are 5 elements together now:
    # 4 regression target fields for class and roi + score for class and roi
    all_roi_info_per_class = tf.concat([rois_reg_targets, reshaped_class_scores], axis=2)
    # Finding transpose so instead of having rows representing rois, we have rows
    # representing classes
    # After this operation, there will be NUMBER_CLASSES rows, each with NUMBER_ROIS elements
    # inside, each with 5 fields inside = 4 reg targets + score for that class and roi
    per_class_rows = tf.transpose(all_roi_info_per_class, perm=[1, 0, 2])

    detected_objects = tf.map_fn(
        lambda x: detect_objects_for_class(rois, x, max_output_size, score_threshold),
        per_class_rows)

    return detected_objects

    # TODO: Prepare output with shape = (# objects detected, # detection fields = 5)


def detect_objects_for_class(rois, class_rois_info, max_output_size, score_threshold):
    # Shape = (# rois, 4 field = x1,y1, x2, y2)
    class_rois_detection_boxes = find_bboxes_from_offsets(rois, class_rois_info)
    class_scores = class_rois_info[:, 4]

    return tf.image.non_max_suppression(
        class_rois_detection_boxes, class_scores,
        max_output_size=max_output_size, score_threshold=score_threshold)


def find_bboxes_from_offsets(rois, rois_reg_targets):
    rois_x = rois[:, 0:1]
    rois_y = rois[:, 1:2]
    rois_w = rois[:, 2:3]
    rois_h = rois[:, 3:4]

    tx = rois_reg_targets[:, 0:1]
    ty = rois_reg_targets[:, 1:2]
    tw = rois_reg_targets[:, 2:3]
    th = rois_reg_targets[:, 3:4]

    gx = tf.add(tf.multiply(rois_w, tx), rois_x)
    gy = tf.add(tf.multiply(rois_h, ty), rois_y)
    gw = tf.multiply(rois_w, tf.exp(tw))
    gh = tf.multiply(rois_h, tf.exp(th))

    x2 = tf.subtract(tf.add(gx, gw), 1)
    y2 = tf.subtract(tf.add(gy, gh), 1)

    # Return [x1, y1, x2, y2] representing the diagonal bbox coordinates
    return tf.concat([gx, gy, x2, y2], axis=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test = detect_and_classify(r, r_scores, r_reg, 4, 0.4)

    # Two rois, two classes, 4 reg fields
    r_reg_test = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                           [[10, 11, 12, 13], [14, 15, 16, 17]]])

    r_scores_test = np.array([[0.1, 0.356, 0.6], [0.3, 0.612, 0.3]])

    r_test = np.array([[0, 0, 5, 5], [5, 5, 5, 5]])

    result = sess.run(test, feed_dict={
        r: r_test,
        r_scores: r_scores_test,
        r_reg: r_reg_test
    })

    print(result)
