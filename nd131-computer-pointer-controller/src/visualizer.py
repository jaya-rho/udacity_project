import cv2

class Visualizer:
    '''

    '''
    def __init__(self):
        None

    def draw_face_bbox(self, frame, bbox_coord, frame_num):
        # for debug, draw a bbox on face
        x_min, y_min, x_max, y_max = bbox_coord
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        cv2.imwrite(f"face_frame_{frame_num}.jpg", frame)
        return frame

    def draw_eye_bbox(self, frame, eye_bbox_coords, frame_num):
        # [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
        [le_x, le_y, re_x, re_y] = eye_bbox_coords
        cv2.rectangle(frame, (le_x - 10, le_y - 10), (le_x + 10, le_y + 10), (0, 0, 255), 3)
        cv2.rectangle(frame, (re_x - 10, re_y - 10), (re_x + 10, re_y + 10), (0, 0, 255), 3)

        cv2.imwrite(f"eye_frame_{frame_num}.jpg", frame)

    def draw_gaze(self, frame, gaze_vector, left_eye_img, right_eye_img, eye_coords):
        x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
        le = cv2.line(left_eye_img, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
        re = cv2.line(right_eye_img, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
        frame[eye_coords[1]:eye_coords[1]+10, eye_coords[0]:eye_coords[0]+10] = le
        frame[eye_coords[3]:eye_coords[3]+10, eye_coords[2]:eye_coords[2]+10] = re
