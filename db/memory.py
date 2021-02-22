

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

    def find_closest_embedding(self, face_embedding, dist_fun):
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

