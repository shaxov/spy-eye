import abc
import cv2
import utils
import typedef


class AbstractTracker:

    @abc.abstractmethod
    def init(self, frame, box):
        """ Initialize tracker with a frame and ROI. """

    @abc.abstractmethod
    def update(self, frame):
        """ Update tracker with new image and return box. """


class FakeTracker(AbstractTracker):

    def init(self, frame, box):
        pass

    def update(self, frame):
        return False, None


class Tracker:

    def __init__(self, tracker_create, trace_len=10):
        self._tracker_create = tracker_create
        self._tracker = self._tracker_create()
        self._trace_len = trace_len

        self.face_id = -1
        self.wasted = False
        self._trace_counter = 0

    def init(self, image, face_box, face_id):
        self._tracker.init(image, face_box)
        self.face_id = face_id

    def update(self, image):
        tacked_ok, box = self._tracker.update(image)
        return tacked_ok, utils.to_int_cords(box)

    def reset(self, image, face_box):
        self._tracker = self._tracker_create()
        self._tracker.init(image, face_box)
        self.reset_trace_counter()

    def is_valid(self):
        return self._trace_counter < self._trace_len and not self.wasted

    def update_trace_counter(self):
        self._trace_counter += 1

    def reset_trace_counter(self):
        self._trace_counter = 0


def get(name, params):
    return Tracker(tracker_create={
        'fake_tracker': FakeTracker,
        'kcf_tracker': cv2.cv2.TrackerKCF_create,
        'mil_tracker': cv2.cv2.TrackerMIL_create,
        'mosse_tracker': cv2.cv2.TrackerMOSSE_create,
    }[name], **params)


def apply(trackers, image, face_boxes):
    out_face_ids, out_face_boxes = [], []
    for tracker in trackers:
        track_ok, tracked_face_box = tracker.update(image)
        if track_ok and tracker.is_valid():
            matched_ok, box_id = utils.match_box(tracked_face_box, face_boxes)
            tracker.update_trace_counter()
            if matched_ok:
                tracked_face_box = face_boxes[box_id]
                tracker.reset(image, tracked_face_box)
                del face_boxes[box_id]
            out_face_ids.append(tracker.face_id)
            out_face_boxes.append(tracked_face_box)
        else:
            tracker.wasted = True
    for face_box in face_boxes:
        out_face_ids.append(typedef.UNKNOWN_FACE_ID)
        out_face_boxes.append(face_box)
    return out_face_ids, out_face_boxes


def drop_wasted(trackers):
    return [tracker for tracker in trackers if not tracker.wasted]


def update_face_ids(trackers, old_face_ids, new_face_ids):
    for tracker in trackers:
        for old_face_id, new_face_id in zip(old_face_ids, new_face_ids):
            if tracker.face_id == old_face_id:
                tracker.face_id = new_face_id
                break
