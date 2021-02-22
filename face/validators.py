import abc
import cv2
import utils
import numpy as np


class AbstractValidator:

    def _is_valid(self, *args, **kwargs):
        return True

    def __call__(self, frame, boxes):
        return list(filter(self._is_valid, boxes))

    @abc.abstractmethod
    def __repr__(self):
        pass


class FakeValidator(AbstractValidator):

    def __call__(self, frame, face_boxes):
        return face_boxes

    @property
    def name(self):
        return "fake_validator"

    def __repr__(self):
        return "<FakeValidator()>"


class MinBoxSizeValidator(AbstractValidator):
    """ Validate bounding boxes by size. """

    def __init__(self, min_width=60, min_height=60):
        self._min_width = min_width
        self._min_height = min_height

    def _is_valid(self, box):
        return box[2] > self._min_width and box[3] > self._min_height

    @property
    def name(self):
        return "min_box_size_validator"

    def __repr__(self):
        return f"<MinBoxSizeValidator(min_width={self._min_width}," \
               f" min_height={self._min_height})>"


class SameDetectionValidator(AbstractValidator):
    """ Validate bounding boxes by size. """

    def __init__(self, max_iou=0.6):
        self._max_iou = max_iou

    @property
    def name(self):
        return "same_detection_validator"

    def __call__(self, frame, boxes):
        start_i = 0
        while True:
            if len(boxes[start_i:]) < 2:
                break
            cur_box = boxes[start_i]
            cur_box_id = start_i
            start_i += 1
            box_id_to_delete = -1
            for box_id, box in enumerate(boxes[start_i:]):
                iou_score = utils.bb_intersection_over_union(cur_box, box)
                if iou_score > self._max_iou:
                    box_id_to_delete = box_id
                    if utils.box_area(cur_box) < utils.box_area(box):
                        box_id_to_delete = cur_box_id
                    break
            if box_id_to_delete != -1:
                del boxes[box_id_to_delete]
                if box_id_to_delete == cur_box_id:
                    start_i -= 1
        return boxes

    def __repr__(self):
        return f"<SameDetectionsValidator(max_iou={self._max_iou})>"


class FFTBlurValidator(AbstractValidator):

    def __init__(self, kernel_size, threshold):
        super().__init__()
        self._kernel_size = kernel_size
        self._threshold = threshold

    def _is_blurred(self, image):
        size = self._kernel_size
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return mean <= self._threshold

    def __call__(self, image, boxes):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [box for box in boxes if not self._is_blurred(utils.crop(gray, *box))]

    def __repr__(self):
        return f"<FFTBlurValidator(kernel_size={self._kernel_size}, threshold={self._threshold})>"

    @property
    def name(self):
        return "fft_blur_validator"


class LaplaceBlurValidator(AbstractValidator):

    def __init__(self, threshold):
        super().__init__()
        self._threshold = threshold

    def _is_blurred(self, image):
        return cv2.Laplacian(image/255, cv2.CV_64F).var() <= self._threshold

    def __call__(self, image, boxes):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [box for box in boxes if not self._is_blurred(utils.crop(gray, *box))]

    def __repr__(self):
        return f"<LaplaceBlurValidator(threshold={self._threshold})>"

    @property
    def name(self):
        return "blur_validator"


def get(name, params=None):
    if name == 'fake_validator':
        return FakeValidator()
    elif name == 'min_box_size_validator':
        return MinBoxSizeValidator(**params)
    elif name == 'same_detection_validator':
        return SameDetectionValidator(**params)
    elif name == 'fft_blur_validator':
        return FFTBlurValidator(**params)
    elif name == 'laplace_blur_validator':
        return LaplaceBlurValidator(**params)
    else:
        raise ValueError(f"Face validator with name '{name}' is not found.")


def get_list(name_params_list):
    return [get(**entry) for entry in name_params_list]


def apply(validators, image, boxes):
    for vld in validators:
        boxes = vld(image, boxes)
    return boxes
