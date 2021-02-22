import db
import cv2
import face
import frame
import yaml
import typedef
import utils
import numpy as np
import copy


def main(config):
    ss = config['source_scale']
    frame_drawer = frame.drawer.Drawer()
    capturer = cv2.VideoCapture(config['source'])
    face_detector = face.detectors.get(**config['face_detector'])
    face_validators = face.validators.get_list(config['face_validators'])
    frame_filters = frame.filters.get_list(config['frame_filters'])
    face_buffer = face.buffer.FaceBuffer(config['face_buffer_size'])
    face_encoder = face.encoders.get(**config['face_encoder'])
    face_recognizer = face.recognizers.get(face_encoder, **config['face_recognizer'])
    storage = db.initialize(**config['database'])
    face_trackers = []

    while True:
        read_ok, image = capturer.read()
        if not read_ok:
            cv2.imshow("Frame", typedef.NO_VIDEO_FRAME)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                capturer.release()
                break
            continue
        image_copy = image.copy()

        image = cv2.resize(image, None, fx=ss, fy=ss, interpolation=cv2.INTER_CUBIC)
        image = frame.filters.apply(frame_filters, image)
        face_boxes = face_detector(image)
        face_boxes = face.validators.apply(face_validators, image, face_boxes)

        face_ids, face_boxes = face.trackers.apply(face_trackers, image, face_boxes)
        face_trackers = face.trackers.drop_wasted(face_trackers)
        _face_boxes = copy.deepcopy(face_boxes)
        face_boxes = face.validators.apply(face_validators, image, face_boxes)
        _face_ids = []
        for face_id, _face_box in zip(face_ids, _face_boxes):
            for face_box in face_boxes:
                if _face_box == face_box:
                    _face_ids.append(face_id)

        for face_id, face_box in zip(_face_ids, face_boxes):

            face_image = utils.crop(image, *face_box)
            face_image = cv2.resize(face_image, tuple(config['face_shape']),
                                    interpolation=cv2.INTER_AREA)

            if face_id == typedef.UNKNOWN_FACE_ID:
                face_id = utils.generate_tmp_face_id()
                tracker = face.trackers.get(**config['face_tracker'])
                tracker.init(image, face_box, face_id)
                face_trackers.append(tracker)

            if utils.is_tmp_id(face_id):
                face_buffer.update(face_id, face_image)
                if face_buffer.is_full(face_id):
                    mean_face = face_buffer.get_mean_face(face_id)
                    recognized_ok, rec_face_id = face_recognizer(mean_face, storage)

                    tracked_face_id = face_id
                    if not recognized_ok:
                        encoded_mean_face = face_encoder(mean_face)
                        face_id = utils.generate_face_id()
                        storage.add(face_id, encoded_mean_face)
                    else:
                        face_id = rec_face_id
                    face.trackers.update_face_ids(face_trackers, [tracked_face_id], [face_id])

            face_box = tuple(np.int64(np.array(face_box) * (1 / ss)))
            frame_drawer.draw_box(image_copy, face_box)
            frame_drawer.draw_face_id(image_copy, face_box, face_id)

        cv2.imshow("Frame", image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            capturer.release()
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        main(yaml.safe_load(file))
