import abc
import cv2
import dlib


class AbstractDetector:

    @abc.abstractmethod
    def __call__(self, frame):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class FakeDetector(AbstractDetector):

    def __call__(self, frame):
        return [(0, 0, 0, 0)]

    @property
    def name(self):
        return "fake_detector"

    def __repr__(self):
        return "<FakeDetector()>"


class CascadeDetector(AbstractDetector):

    def __init__(self, path_to_xml='face/files/haarcascade_frontalface_default.xml',
                 scale_factor=1.2, min_neighbour=12):
        self._face_cascade = cv2.CascadeClassifier(path_to_xml)
        self._scale_factor = scale_factor
        self._min_neighbour = min_neighbour

    def __call__(self, frame):
        boxes = self._face_cascade.detectMultiScale(
            frame, self._scale_factor, self._min_neighbour)
        return [tuple(box) for box in boxes]

    @property
    def name(self):
        return "cascade_detector"

    def __repr__(self):
        return f"<CascadeDetector(scale_factor={self._scale_factor}," \
               f" min_neighbour={self._min_neighbour})>"


class HOGDetector(AbstractDetector):

    def __init__(self, threshold=0.5):
        self._threshold = threshold
        self._detector = dlib.get_frontal_face_detector()

    def __call__(self, frame):
        faces, scores, idx = self._detector.run(frame, 1, -1)
        return [
            (rect.left(),
             rect.top(),
             rect.right() - rect.left(),
             rect.bottom() - rect.top())
            for score, rect in zip(scores, faces)
            if score > self._threshold
        ]

    @property
    def name(self):
        return "hog_detector"

    def __repr__(self):
        return f"<HOGDetector(threshold={self._threshold})>"


def get(name, params):
    if name == 'fake_detector':
        return FakeDetector()
    elif name == 'cascade_detector':
        return CascadeDetector(**params)
    elif name == 'hog_detector':
        return HOGDetector(**params)
    else:
        raise ValueError(f"Face detector with name '{name}' is not found.")
