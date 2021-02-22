import abc
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
            face_id, score = storage.find_closest_embedding(face_embedding, utils.euc_dist)
            recognized_ok = score < self._threshold
        return recognized_ok, face_id

    def __repr__(self):
        return f"<DLibRecognizer(encoder={self._encoder}, threshold={self._threshold})>"




# class SVDRecognizer(AbstractRecognizer):
#
#     @staticmethod
#     def _norm_log_prob(svs, mu, sigma):
#         p = 1. / (2 * np.pi * (sigma + 1e-16)) * np.exp(
#             -0.5 * ((svs - mu) / (sigma + 1e-16)) ** 2) + 1e-16
#         return np.sum(np.log(p))
#
#     @staticmethod
#     def _is_in_trust_int(log_prob, mu, sigma, size, alpha=2.576):
#         min_log_prob = SVDRecognizer._norm_log_prob(
#             mu - alpha * (sigma / np.sqrt(size)), mu, sigma)
#         return log_prob > min_log_prob
#
#     def recognize_face(self, face_image):
#         if self._storage.is_empty():
#             return False, typedef.UNKNOWN_FACE_ID
#         min_norm_diff = np.inf
#         rec_face_id = typedef.UNKNOWN_FACE_ID
#         record = None
#         recognized_ok = False
#         for face_id in self._storage:
#             record = self._storage[face_id]
#             svs = utils.orthogonal_projections(
#                 face_image, **record['uv_mats'])
#             norm_diff = np.linalg.norm(svs - record['dist_params']['mu'])
#             if norm_diff < 4.:
#                 recognized_ok = True
#                 if norm_diff < min_norm_diff:
#                     min_norm_diff = norm_diff
#                     rec_face_id = face_id

            # log_prob = SVDRecognizer._norm_log_prob(
            #     svs, mu=record['dist_params']['mu'], sigma=record['dist_params']['sigma'])
            # if log_prob > max_log_prob:
            #     max_log_prob = log_prob
            #     rec_face_id = face_id
            #     recognized_ok = True
        # if record is not None:
        #     if not self._is_in_trust_int(
        #             max_log_prob, **record['dist_params']):
        #         rec_face_id = typedef.UNKNOWN_FACE_ID
        #         recognized_ok = False
        # return recognized_ok, rec_face_id


def get(encoder, name, params):
    if name == "face_recognizer":
        return FakeRecognizer()
    elif name == "dlib_recognizer":
        return DLibRecognizer(encoder, **params)
    else:
        raise ValueError(f"Face recognizer with name '{name}' is not found.")
