import cv2


class Drawer:

    @staticmethod
    def draw_box(frame, box, color=(0, 255, 0)):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    @staticmethod
    def draw_face_id(frame, box, face_id):
        x, y, w, h = box
        pt = (x + 2, y + h + 20)
        cv2.putText(frame, f"ID: {face_id}", pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)