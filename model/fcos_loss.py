import tensorflow as tf

from model.fcos_head import compute_locations
from model.focal_loss import focal_loss_sigmoid
from model.iou_loss import iou_loss

INF = 100000000

def bbox_regress_target(bboxes, points):
    '''
    Args:
        bboxes: [batch_size, m, 4+]
        points: [n, 2]
    Returns:
        reg_target: [batch_size, n, m, 4]
    '''
    xs = points[:, 0][tf.newaxis, :, tf.newaxis]
    ys = points[:, 1][tf.newaxis, :, tf.newaxis]
    l = xs - tf.expand_dims(bboxes[:, :, 0], 1)
    t = ys - tf.expand_dims(bboxes[:, :, 1], 1)
    r = tf.expand_dims(bboxes[:, :, 2], 1) - xs
    b = tf.expand_dims(bboxes[:, :, 3], 1) - ys
    reg_target = tf.stack([l, t, r, b], -1)
    return reg_target


def bbox_area(bboxes):
    '''
    Args:
        bboxes: [batch_size, m, 4+]
    Returns:
        area: [batch_size, m]
    '''
    width = bboxes[:, :, 2] - bboxes[:, :, 0] + 1
    height = bboxes[:, :, 3] - bboxes[:, :, 1] + 1
    return width * height


def compute_targets_for_locations(locations, gt_boxes, object_sizes_of_interest):
    '''
    Args:
        locations: [n, 2]
        gt_boxes: [batch_size, m, 4]
        object_sizes_of_interest: List[2]
    Returns:
        cls_target: [batch_size, n]
        reg_target: [batch_size, n, 4]
    '''
    n = tf.shape(locations)[0]
    m = tf.shape(gt_boxes)[1]
    batch_size = tf.shape(gt_boxes)[0]

    reg_target = bbox_regress_target(gt_boxes, locations)   # [batch_size, n, m, 4]

    is_in_boxes = tf.reduce_min(reg_target, 3) > 0  # [batch_size, n, m]
    
    # limit the regression range for each location
    max_reg_target = tf.reduce_max(reg_target, 3)   # [batch_size, n, m]
    is_cared_in_the_level = tf.logical_and(
        (max_reg_target >= object_sizes_of_interest[0]),
        (max_reg_target <= object_sizes_of_interest[1])
    ) # [batch_size, n, m]

    # 
    is_valid = tf.broadcast_to(gt_boxes[:, tf.newaxis, :, 4] > 0, [batch_size, n, m])

    area = bbox_area(gt_boxes) # [batch_size, m]
    locations_to_gt_area = tf.broadcast_to(area[:, tf.newaxis, :], [batch_size, n, m])
    locations_to_gt_area = tf.where(
        (is_in_boxes & is_cared_in_the_level & is_valid),
        locations_to_gt_area,
        tf.constant(INF, dtype=tf.float32)
    )   # TODO mask assignment

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    locations_to_gt_inds = tf.argmin(locations_to_gt_area, 2) # [batch_size, n]
    locations_to_min_area = tf.reduce_min(locations_to_gt_area, 2) # [batch_size, n]

    reg_target = tf.gather_nd(
        reg_target, 
        locations_to_gt_inds[:, :, tf.newaxis],
        batch_dims=2)
    cls_target = tf.gather_nd(
        gt_boxes[:, :, 4],
        locations_to_gt_inds[:, :, tf.newaxis],
        batch_dims=1)
    cls_target = tf.where(
        locations_to_min_area < tf.constant(INF-1, dtype=tf.float32),
        cls_target,
        tf.constant(0, dtype=tf.float32)
    )   # TODO mask assignment

    return cls_target, reg_target


def prepare_targets(points, gt_boxes):
    '''
        points: List[Tensor]
    '''
    object_sizes_of_interest = [
        [0, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, INF],
    ]
    cls_targets = []
    reg_targets = []
    for lvl, points_lvl in enumerate(points):
        cls_target_lvl, reg_target_lvl = compute_targets_for_locations(
            points_lvl, gt_boxes, object_sizes_of_interest[lvl]
        )
        #print("done!")
        cls_targets.append(cls_target_lvl)
        reg_targets.append(reg_target_lvl)
        reg_target_lvl = tf.reduce_min(reg_target_lvl, 2)
        #print(reg_target_lvl[cls_target_lvl>0])
    return cls_targets, reg_targets


def compute_centerness_targets(reg_targets):
    left_right_min = tf.minimum(reg_targets[:, 0], reg_targets[:, 2])
    left_right_max = tf.maximum(reg_targets[:, 0], reg_targets[:, 2])
    top_bottom_min = tf.minimum(reg_targets[:, 1], reg_targets[:, 3])
    top_bottom_max = tf.maximum(reg_targets[:, 1], reg_targets[:, 3])
    centerness = (left_right_min/left_right_max) * (top_bottom_min/top_bottom_max)
    return tf.sqrt(centerness)


def fcos_loss(box_cls,
            box_regression,
            centerness,
            gt_boxes):
    locations = []
    num_classes = tf.shape(box_cls[0])[-1]
    batch_size = tf.shape(box_cls[0])[0]

    #TODO compute locations 
    strides = [8, 16, 32, 64, 128]
    for lvl, stride in enumerate(strides):
        locations.append(compute_locations(box_cls[lvl], stride))
    
    cls_targets, reg_targets = prepare_targets(locations, gt_boxes)
    
    # concat over level 
    box_cls_flatten = []
    box_regression_flatten = []
    centerness_flatten = []
    cls_ind_flatten = []
    reg_targets_flatten = []
    for l in range(len(cls_targets)):
        box_cls_flatten.append(tf.reshape(box_cls[l], [-1, num_classes]))
        box_regression_flatten.append(tf.reshape(box_regression[l], [-1, 4]))
        cls_ind_flatten.append(tf.reshape(cls_targets[l], [-1]))
        reg_targets_flatten.append(tf.reshape(reg_targets[l], [-1, 4]))
        centerness_flatten.append(tf.reshape(centerness[l], [-1]))
    box_cls_flatten = tf.concat(box_cls_flatten, 0)
    box_regression_flatten = tf.concat(box_regression_flatten, 0)
    centerness_flatten = tf.concat(centerness_flatten, 0)
    cls_ind_flatten = tf.concat(cls_ind_flatten, 0)
    reg_targets_flatten = tf.concat(reg_targets_flatten,0)
    
    pos_inds = tf.where(cls_ind_flatten>0)

    # cls loss
    onehot_cls_target = tf.equal(
        tf.range(1, num_classes+1, dtype=tf.int32)[tf.newaxis, :],
        tf.cast(cls_ind_flatten[:, tf.newaxis], tf.int32)
    )
    onehot_cls_target = tf.cast(onehot_cls_target, tf.float32)
    cls_loss = focal_loss_sigmoid(
        box_cls_flatten,
        onehot_cls_target
    ) / tf.cast((tf.shape(pos_inds)[0] + batch_size), tf.float32)  # add batch_size to avoid dividing by a zero

    box_regression_flatten = tf.gather_nd(box_regression_flatten, pos_inds)
    reg_targets_flatten = tf.gather_nd(reg_targets_flatten, pos_inds)
    centerness_flatten = tf.gather_nd(centerness_flatten, pos_inds)

    # centerness loss
    centerness_targets = compute_centerness_targets(reg_targets_flatten)
    centerness_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        centerness_flatten,
        centerness_targets
    )
    centerness_loss = tf.reduce_mean(centerness_loss)

    # regression loss
    reg_loss = iou_loss(
        box_regression_flatten,
        reg_targets_flatten,
        centerness_targets
    )

    return cls_loss, centerness_loss, reg_loss
