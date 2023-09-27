import numpy as np


def get_max_box(boxes, scores, classes):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    max_index = np.argmax(areas)
    return boxes[max_index], scores[max_index], classes[max_index]


def get_square_coordinates(box, img_shape):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1) * 1.25
    side = min(int(side), img_shape[1], img_shape[0])
    x1, y1, x2, y2 = (
        int(center_x - side // 2),
        int(center_y - side // 2),
        int(center_x + side // 2),
        int(center_y + side // 2),
    )
    return x1, y1, x2, y2


def adjust_boundaries(x1, y1, x2, y2, img_shape):
    left_pad = max(-x1, 0)
    top_pad = max(-y1, 0)
    right_pad = max(x2 - img_shape[1], 0)
    bottom_pad = max(y2 - img_shape[0], 0)

    x1 += left_pad
    y1 += top_pad
    x2 -= right_pad
    y2 -= bottom_pad

    return x1, y1, x2, y2, top_pad, bottom_pad, left_pad, right_pad
