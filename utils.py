import dlib
import numpy as np


def orthogonal_projections(matrix, u, vh):
    return np.diag(u.T @ matrix @ vh.T)


def svd(image):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    return {'u': u, 's': s, 'vh': vh}


def crop(image, x, y, w, h):
    return image[y:y + h, x:x + w]


# def crop_dlib(image, x, y, w, h):
#     left = x
#     top = y
#     right = x + w
#     bottom = y + h
#     rect = dlib.rectangle(left, top, right, bottom)
#     return dlib.get_face_chip(image, rect)


def _face_id_generator_func():
    face_id = 1
    while True:
        yield f"{face_id}"
        face_id += 1


def _tmp_face_id_generator_func():
    face_id = 1
    while True:
        yield f"tmp:{face_id}"
        face_id += 1


_face_id_generator = _face_id_generator_func()
_tmp_face_id_generator = _tmp_face_id_generator_func()


def generate_tmp_face_id():
    return next(_tmp_face_id_generator)


def generate_face_id():
    return next(_face_id_generator)


def is_tmp_id(face_id):
    return face_id.split(':')[0] == "tmp"


def box_area(box):
    return box[2] * box[3]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def match_box(face_box, face_boxes, threshold=0.3):
    if not face_boxes:
        return False, 0
    max_iou = 0
    best_face_id = -1
    for _face_id, _face_box in enumerate(face_boxes):
        iou = bb_intersection_over_union(face_box, _face_box)
        if iou > max_iou:
            max_iou = iou
            best_face_id = _face_id
    matched_ok = max_iou >= threshold
    return matched_ok, best_face_id


def to_int_cords(face_box):
    return tuple([int(cord) for cord in face_box])


def euc_dist(vec1, vec2, p=2, weights=1):
    return np.linalg.norm(weights*(vec1 - vec2), p)


def cos_dist(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
