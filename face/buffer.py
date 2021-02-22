import numpy as np


class FaceBufferConfig:
    max_len = 25


class BufferUpdateException(Exception):

    def __init__(self):
        self.message = "Update is not possible. Max size " \
                       "of mean face buffer is achieved."
        super().__init__(self.message)


class FaceBuffer:

    def __init__(self, max_len):
        self._max_len = max_len
        self._buffer = {}

    def update(self, face_id, face_image):
        if self.is_full(face_id):
            raise BufferUpdateException()

        if face_id not in self._buffer:
            self._buffer[face_id] = []
        self._buffer[face_id].append(face_image)

    def is_full(self, face_id):
        if face_id not in self._buffer:
            return False
        return len(self._buffer[face_id]) == self._max_len

    def get_mean_face(self, face_id):
        return np.mean(self._buffer.get(face_id), axis=0).astype(np.uint8)

    def get_all_faces(self, face_id):
        return self._buffer[face_id]

    def drop_face(self, face_id):
        del self._buffer[face_id]
