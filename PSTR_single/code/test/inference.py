import numpy as np
import math
from copy import deepcopy

ORANGE = (0, 165, 255)

template = [
        [0, 0, 0, 0, 1], # side
        [1, 1, 0, 0, 1], # left / right side

        [0, 0, 1, 0, 0], # tail
        [1, 1, 1, 0, 0], # tail left / right side

        [0, 0, 0, 1, 0], # head
        [1, 1, 0, 1, 0], # head left / right side

        [1, 0, 0, 0, 1], # side
        [0, 1, 0, 0, 1], # side

        [1, 0, 1, 0, 0], # tail left / right side
        [0, 1, 1, 0, 0], # tail left / right side

        [1, 0, 0, 1, 0], # head left / right side
        [0, 1, 0, 1, 0], # head left / right side

        [1, 1, 0, 0, 0], # left / right side
        [1, 0, 0, 0, 0], # side
        [0, 1, 0, 0, 0], # side
        [0, 0, 0, 0, 0], # side
    ]

tpid2pose = {
    0: 'side',
    1: 'lrside',
    2: 'tail',
    3: 'taillrside',
    4: 'head',
    5: 'headlrside',
    6: 'lrside',
    7: 'lrside',
    8: 'taillrside',
    9: 'taillrside',
    10: 'headlrside',
    11: 'headlrside',
    12 : 'lrside',
    13: 'side',
    14: 'side',
    15: 'side',
}

up = 0.2
down = 0.1
edge = 0.05
side_color = ORANGE
num_joints = 5


def tbox(detbox, joints, scores, strange_obj=None, strange_pose=None):

    xmin, ymin, xmax, ymax = detbox

    # Record raw joint detections
    ori_joints, ori_scores = deepcopy(joints), deepcopy(scores)

    # # TODO: [Rule 1] Select the key point with the largest score among j_3, j_4, j_5
    # facade_scores = np.zeros([num_joints])
    # facade_scores[2:] = scores[2:]
    # max_facade_idx = np.argmax(facade_scores)
    # if max_facade_idx <= 1:
    #     continue
    # facade_scores_temp = np.ones([num_joints]) * -1.
    # facade_scores_temp[max_facade_idx] = facade_scores[max_facade_idx]
    # scores[2:] = facade_scores_temp[2:]

    # Control the up and bottom boundaries for j_1 and j_2
    joints[0][1] = np.minimum(np.maximum(joints[0][1], int(ymin)), int(ymax))
    joints[1][1] = np.minimum(np.maximum(joints[1][1], int(ymin)), int(ymax))

    # Control the up boundary for j_3, j_4 and j_5
    joints[2][1] = ymax  # align to bbox
    joints[3][1] = ymax  # align to bbox
    joints[4][1] = ymax  # align to bbox

    # Control the left and right boundaries for j_3, j_4 and j_5
    joints[2][0] = np.minimum(np.maximum(joints[2][0], int(xmin)), int(xmax))
    joints[3][0] = np.minimum(np.maximum(joints[3][0], int(xmin)), int(xmax))
    joints[4][0] = np.minimum(np.maximum(joints[4][0], int(xmin)), int(xmax))

    # TODO: [Rule 2] Drop detected j_1 and j_2 when they are located on different sides
    if abs(joints[0][0] - joints[1][0]) > (xmax - xmin) / 2:  # if 0, 1 points different side
        joints[0][:] = None
        joints[1][:] = None
        scores[0] = -1.
        scores[1] = -1.

    # TODO: [Rule 3] First pose encoding
    # joints[scores == -1] = None
    scores[scores < 0.01] = -1.
    joints[scores == -1] = None
    pose_idx = [1] * num_joints
    for k in range(num_joints):
        if math.isnan(joints[k][0]):
            pose_idx[k] = 0

    # Record strange pose instance
    if not pose_idx in template:
        print("pose_idx: {} not in template".format(pose_idx))
        print('ori_joints:\n{}'.format(ori_joints))
        print('ori_scores:\n{}'.format(ori_scores))
        strange = True
        # raise ValueError('invalid pose_idx: {}'.format(pose_idx))
        # image = draw_sticks(image, joints, edges, side_color, line_width)
        # image = draw_joints(image, joints, marker_size, scores)
        strange_obj += 1
        if pose_idx not in strange_pose:
            strange_pose.append(pose_idx)

    # TODO: [Rule 4] Determine [Left] and [Right] firstly
    pose = None
    LorR = None
    for tp in range(len(template)):
        if pose_idx == template[tp]:
            pose = tpid2pose[tp]
            if tp in [1, 3, 5, 6, 7, 8, 9, 10, 11, 12]:  # need to classify left or right
                # If both j_1 and j_2 are not nan
                if not math.isnan(joints[0][0]) and not math.isnan(joints[1][0]):
                    x_p0 = joints[0][0]  # j_1_x
                    x_p1 = joints[1][0]  # j_2_x
                    d_x_p0_xmin = abs(x_p0 - xmin)  # j_1_x to xmin
                    d_x_p0_xmax = abs(x_p0 - xmax)  # j_1_x to xmax
                    d_x_p1_xmin = abs(x_p1 - xmin)  # j_2_x to xmin
                    d_x_p1_xmax = abs(x_p1 - xmax)  # j_2_x to xmax

                    if d_x_p0_xmin < d_x_p0_xmax and d_x_p1_xmin < d_x_p1_xmax:
                        LorR = 'left'
                    elif d_x_p0_xmin > d_x_p0_xmax and d_x_p1_xmin > d_x_p1_xmax:
                        LorR = 'right'
                    else:
                        print('two side points not the same side')
                        print('from j_1_x to xmin | j_1_x to xmax: {} | {}'.format(d_x_p0_xmin, d_x_p0_xmax))
                        print('from j_2_x to xmin | j_2_x to xmax: {} | {}'.format(d_x_p1_xmin, d_x_p1_xmax))
                        # image = draw_sticks(image, joints, edges, side_color, line_width)
                        # image = draw_joints(image, joints, marker_size, scores)
                        # raise ValueError("=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                        print("=> pose: {}, LorR: {} joints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                        max_id = np.argmax(scores)
                        if max_id == 0 or max_id == 1 or max_id == 4:
                            pose = 'side'
                        elif max_id == 2:
                            pose = 'tail'
                        elif max_id == 3:
                            pose = 'head'

                # If j_1 is not nan and j_2 is nan
                elif not math.isnan(joints[0][0]) and math.isnan(joints[1][0]):
                    x_p0 = joints[0][0]
                    d_x_p0_xmin = abs(x_p0 - xmin)  # From j_1_x to xmin
                    d_x_p0_xmax = abs(x_p0 - xmax)  # From j_1_x to xmax
                    # Reconstruct j_2 use its raw detection
                    joints[1] = deepcopy(ori_joints[1])
                    if d_x_p0_xmin < d_x_p0_xmax:
                        LorR = 'left'
                    elif d_x_p0_xmin > d_x_p0_xmax:
                        LorR = 'right'
                    else:
                        raise ValueError(
                            "=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))

                # If j_1 is nan and j_2 is not nan
                elif math.isnan(joints[0][0]) and not math.isnan(joints[1][0]):
                    x_p1 = joints[1][0]
                    d_x_p1_xmin = abs(x_p1 - xmin)  # From j_2_x to xmin
                    d_x_p1_xmax = abs(x_p1 - xmax)  # From j_2_x to xmax
                    # Reconstruct j_1 use its raw detection
                    joints[0] = deepcopy(ori_joints[0])
                    if d_x_p1_xmin < d_x_p1_xmax:
                        LorR = 'left'
                    elif d_x_p1_xmin > d_x_p1_xmax:
                        LorR = 'right'
                    else:
                        raise ValueError(
                            "=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                else:
                    raise ValueError("Both j_1 and j_2 are nan\n"
                                     "=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints,
                                                                                             scores))

    # Give no pose currently
    if pose == None:
        print("[STRANGE!] this pose = None\n=> pose_idx: {}\n=> pose: {}, LorR: {}\njoints:\n{}scores:\n{}".format(
            pose_idx, pose, LorR, joints, scores))
        strange = True

    # TODO: [Rule 5] Align j_1 and j_2 on the left/right edge based on the determined LorR
    if LorR == 'left':
        joints[0][0] = int(xmin)
        joints[1][0] = int(xmin)

    elif LorR == 'right':
        joints[0][0] = int(xmax)
        joints[1][0] = int(xmax)

    # TODO: [Rule 6] Construct final TBox
    pose_name = None
    if pose == 'side':
        pose_name = 'side'

    elif pose == 'tail':
        pose_name = 'tail'

    elif pose == 'head':
        pose_name = 'head'

    elif pose == 'lrside':
        if LorR == 'left':
            if abs(xmax - 1280) < abs(xmax - xmin) * 0.2:  # True left side
                pose_name = 'left_side'
            else:
                pose_name = 'side'
        elif LorR == 'right':
            if abs(xmin - 0) < abs(xmax - xmin) * 0.2:  # True right side
                pose_name = 'right_side'
            else:
                pose_name = 'side'
        else:
            raise ValueError("=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))

    elif pose == 'taillrside':
        if LorR == 'left':
            pose_name = 'tail_left_side'
            if abs(joints[2][0] - xmax) < edge * abs(xmax - xmin) or abs(joints[2][0] - xmin) < edge * abs(xmax - xmin):
                ts_id = np.argmax(scores)
                # image = draw_sticks(image, joints, edges, side_color, line_width)
                # image = draw_joints(image, joints, marker_size, scores)
                print('box: {}'.format([xmin, ymin, xmax, ymax]))
                print('p2 is too close to box boundary')
                print("=> pose: {}, LorR: {}\n"
                      "joints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                if ts_id == 0 or ts_id == 1:
                    pose_name = 'side'
                elif ts_id == 2:
                    pose_name = 'tail'
                else:
                    raise ValueError('pose: {}, LorR: {}, ts_id: {}\nscores:\n{}'.format(pose, LorR, ts_id, scores))

        elif LorR == 'right':
            pose_name = 'tail_right_side'
            if abs(joints[2][0] - xmax) < edge * abs(xmax - xmin) or abs(joints[2][0] - xmin) < edge * abs(xmax - xmin):
                ts_id = np.argmax(scores)
                # image = draw_sticks(image, joints, edges, side_color, line_width)
                # image = draw_joints(image, joints, marker_size, scores)
                print('box: {}'.format([xmin, ymin, xmax, ymax]))
                print('p2 is too close to box boundary')
                print("=> pose: {}, LorR: {}\n"
                      "joints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                if ts_id == 0 or ts_id == 1:
                    pose_name = 'side'
                elif ts_id == 2:
                    pose_name = 'tail'
                else:
                    raise ValueError('pose: {}, LorR: {}, ts_id: {}\nscores:\n{}'.format(pose, LorR, ts_id, scores))
        else:
            raise ValueError("=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))


    elif pose == 'headlrside':
        if LorR == 'left':
            pose_name = 'head_left_side'
            if abs(joints[3][0] - xmax) < edge * abs(xmax - xmin) \
                    or abs(joints[3][0] - xmin) < edge * abs(xmax - xmin):
                ts_id = np.argmax(scores)
                # image = draw_sticks(image, joints, edges, side_color, line_width)
                # image = draw_joints(image, joints, marker_size, scores)
                print('p3 is too close to box boundary')
                print('box: {}'.format([xmin, ymin, xmax, ymax]))
                print("=> pose: {}, LorR: {}\n"
                      "joints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                if ts_id == 0 or ts_id == 1:
                    pose_name = 'side'
                elif ts_id == 3:
                    pose_name = 'head'
                else:
                    raise ValueError('pose: {}, LorR: {}, ts_id: {}\nscores:\n{}'.format(pose, LorR, ts_id, scores))


        elif LorR == 'right':
            pose_name = 'head_right_side'
            if abs(joints[3][0] - xmax) < edge * abs(xmax - xmin) \
                    or abs(joints[3][0] - xmin) < edge * abs(xmax - xmin):
                ts_id = np.argmax(scores)
                # image = draw_sticks(image, joints, edges, side_color, line_width)
                # image = draw_joints(image, joints, marker_size, scores)
                print('p3 is too close to box boundary')
                print('box: {}'.format([xmin, ymin, xmax, ymax]))
                print("=> pose: {}, LorR: {}\n"
                      "joints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))
                if ts_id == 0 or ts_id == 1:
                    pose_name = 'side'
                elif ts_id == 3:
                    pose_name = 'head'
                else:
                    raise ValueError('pose: {}, LorR: {}, ts_id: {}\nscores:\n{}'.format(pose, LorR, ts_id, scores))
        else:
            raise ValueError("=> pose: {}, LorR: {}\njoints:\n{}\nscores\n{}".format(pose, LorR, joints, scores))

    return detbox, joints, scores, pose_name