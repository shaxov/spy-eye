import numpy as np
import utils


class MemoryDB:

    def __init__(self):
        self._storage = {}

    def add(self, face_id, face_embedding):
        self._storage[face_id] = face_embedding

    def get_face(self, face_id):
        return self._storage[face_id]

    def get_face_ids(self):
        return list(self._storage.keys())

    def is_empty(self):
        return not self._storage

    def __contains__(self, face_id):
        return face_id in self._storage

    def find_closest_by_dlib_embedding(self, face_embedding, dist_fun):
        if self.is_empty():
            raise ValueError("Search is impossible. Storage is empty.")

        min_dist = float('inf')
        min_dist_face_id = -1
        for face_id in self._storage:
            dist = dist_fun(face_embedding, self._storage[face_id])
            if dist < min_dist:
                min_dist = dist
                min_dist_face_id = face_id
        return min_dist_face_id, min_dist

    def find_closest_by_svd_embedding(self, face_image):
        if self.is_empty():
            raise ValueError("Search is impossible. Storage is empty.")

        min_dist = float('inf')
        min_dist_face_id = -1
        min_dlib_embedding = None
        for face_id in self._storage:
            dlib_embedding, usv_mats = self._storage[face_id]
            svd_embedding = utils.orthogonal_projections(
                face_image, usv_mats['u'], usv_mats['vh'])
            dist = utils.euc_dist(svd_embedding, usv_mats['s'])
            if dist < min_dist:
                min_dist = dist
                min_dist_face_id = face_id
                min_dlib_embedding = dlib_embedding
        return min_dist_face_id, min_dlib_embedding

    def generate_face_id(self):
        return str(len(self._storage) + 1)
