import abc
import cv2
import utils
import typedef
import numpy as np


class AbstractRecognizer:

    @abc.abstractmethod
    def __call__(self, face_image, storage):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class FakeRecognizer(AbstractRecognizer):

    def __call__(self, face_image, storage):
        return True, typedef.FAKE_FACE_ID

    def __repr__(self):
        return "<FaceRecognizer()>"


class DLibRecognizer(AbstractRecognizer):

    def __init__(self, encoder, threshold=0.6):
        super().__init__()
        self._encoder = encoder
        self._threshold = threshold

    def __call__(self, face_image, storage):
        face_id = typedef.UNKNOWN_FACE_ID
        recognized_ok = False
        if not storage.is_empty():
            face_embedding = self._encoder(face_image)
            face_id, score = storage.find_closest_by_dlib_embedding(
                face_embedding, utils.euc_dist)
            recognized_ok = score < self._threshold
        return recognized_ok, face_id

    def __repr__(self):
        return f"<DLibRecognizer(encoder={self._encoder}, threshold={self._threshold})>"


class DlibSVDRecognizer(DLibRecognizer):

    def __call__(self, face_image, storage):
        face_id = typedef.UNKNOWN_FACE_ID
        recognized_ok = False
        if not storage.is_empty():
            face_id, dlib_embedding = storage.find_closest_by_svd_embedding(
                cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY))
            score = utils.euc_dist(self._encoder(face_image, svd=False), dlib_embedding)
            recognized_ok = score < self._threshold
        return recognized_ok, face_id

    def __repr__(self):
        return f"<DLibSVDRecognizer(encoder={self._encoder}, threshold={self._threshold})>"


def get(encoder, name, params):
    if name == "face_recognizer":
        return FakeRecognizer()
    elif name == "dlib_recognizer":
        return DLibRecognizer(encoder, **params)
    elif name == "dlib_svd_recognizer":
        return DlibSVDRecognizer(encoder, **params)
    else:
        raise ValueError(f"Face recognizer with name '{name}' is not found.")
