import abc
import cv2
import dlib
import utils
import numpy as np


class AbstractEncoder:

    @abc.abstractmethod
    def __call__(self, face_image):
        """ Encode face image to some vector/matrix representation. """

    @abc.abstractmethod
    def __repr__(self):
        pass


class FakeEncoder(AbstractEncoder):

    def __call__(self, face_image):
        return 0.

    def __repr__(self):
        return "<FakeEncoder()>"


class DlibEncoder(AbstractEncoder):

    def __init__(self, path_to_cnn_model, path_to_landmark_model):
        self._face_encoder = dlib.face_recognition_model_v1(path_to_cnn_model)
        self._sp = dlib.shape_predictor(path_to_landmark_model)

    def __call__(self, face_image):
        shape = self._sp(face_image, dlib.rectangle(0, 0, *face_image.shape[:-1]))
        return np.array(self._face_encoder.compute_face_descriptor(face_image, shape))

    def __repr__(self):
        return "<DlibEncoder()>"


class DlibSVDEncoder(DlibEncoder):

    def __call__(self, face_image, svd=True):
        out = super().__call__(face_image)
        if svd:
            out = (out, utils.svd(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)))
        return out

    def __repr__(self):
        return "<DlibSVDEncoder()>"


def get(name, params):
    if name == "fake_encoder":
        return FakeEncoder()
    elif name == "dlib_encoder":
        return DlibEncoder(**params)
    elif name == "dlib_svd_encoder":
        return DlibSVDEncoder(**params)
    else:
        raise ValueError(f"Encoder with name '{name}' is not found.")
