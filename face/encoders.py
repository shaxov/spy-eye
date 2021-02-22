import abc
import dlib
import utils
import numpy as np


class AbstractEncoder:

    @abc.abstractmethod
    def __call__(self, face_image):
        """ Encode face image to some vector/matrix representation. """


class FakeEncoder(AbstractEncoder):

    def __call__(self, face_image):
        return 0.


class DlibEncoder(AbstractEncoder):

    def __init__(self, path_to_cnn_model, path_to_landmark_model):
        self._face_encoder = dlib.face_recognition_model_v1(path_to_cnn_model)
        self._sp = dlib.shape_predictor(path_to_landmark_model)

    def __call__(self, face_image):
        shape = self._sp(face_image, dlib.rectangle(0, 0, *face_image.shape[:-1]))
        return np.array(self._face_encoder.compute_face_descriptor(face_image, shape))

# class SVDEncoder(AbstractEncoder):
#
#     def encode_face(self, face_image):
#         return utils.svd(face_image)
#
#     def encode_face_buffer(self, face_buffer):
#         mean_face = np.mean(face_buffer, axis=0)
#         uv_mats = utils.svd(mean_face)
#         embeddings = [
#             utils.orthogonal_projections(face_image, **uv_mats)
#             for face_image in face_buffer
#         ]
#         return {
#             "uv_mats": uv_mats,
#             "dist_params": {
#                 "mu": np.mean(embeddings, axis=0),
#                 "sigma": np.std(embeddings, axis=0),
#                 "size": len(embeddings),
#             }
#         }


def get(name, params):
    if name == "fake_encoder":
        return FakeEncoder()
    elif name == "dlib_encoder":
        return DlibEncoder(**params)
    else:
        raise ValueError(f"Encoder with name '{name}' is not found.")
