import cv2
import numpy as np
import math

class Visualizer:
    '''

    '''
    def __init__(self):
        None

    def draw_face_bbox(self, frame, bbox_coord, frame_num):
        # for debug, draw a bbox on face
        x_min, y_min, x_max, y_max = bbox_coord
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        cv2.imwrite(f"output_vis/face_frame_{frame_num}.jpg", frame)
        return frame

    def draw_eye_bbox(self, frame, eye_bbox_coords, frame_num):
        # [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
        [le_x, le_y, re_x, re_y] = eye_bbox_coords
        cv2.rectangle(frame, (le_x - 10, le_y - 10), (le_x + 10, le_y + 10), (0, 0, 255), 3)
        cv2.rectangle(frame, (re_x - 10, re_y - 10), (re_x + 10, re_y + 10), (0, 0, 255), 3)

        cv2.imwrite(f"output_vis/eye_frame_{frame_num}.jpg", frame)
        return frame

    def draw_head_pose(self, frame, hp_out):

        # the below source codes from https://knowledge.udacity.com/questions/171017
        def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
            yaw *= np.pi / 180.0
            pitch *= np.pi / 180.0
            roll *= np.pi / 180.0
            cx = int(center_of_face[0])
            cy = int(center_of_face[1])
            r_x = np.array([[1, 0, 0],
                            [0, math.cos(pitch), -math.sin(pitch)],
                            [0, math.sin(pitch), math.cos(pitch)]])
            r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                            [0, 1, 0],
                            [math.sin(yaw), 0, math.cos(yaw)]])
            r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                            [math.sin(roll), math.cos(roll), 0],
                            [0, 0, 1]])
            r = r_z @ r_y @ r_x

            def build_camera_matrix(center_of_face, focal_length):
                cx = int(center_of_face[0])
                cy = int(center_of_face[1])
                camera_matrix = np.zeros((3, 3), dtype='float32')
                camera_matrix[0][0] = focal_length
                camera_matrix[0][2] = cx
                camera_matrix[1][1] = focal_length
                camera_matrix[1][2] = cy
                camera_matrix[2][2] = 1
                return camera_matrix

            camera_matrix = build_camera_matrix(center_of_face, focal_length)
            xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
            yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
            zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
            zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
            o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
            o[2] = camera_matrix[0][0]
            xaxis = np.dot(r, xaxis) + o
            yaxis = np.dot(r, yaxis) + o
            zaxis = np.dot(r, zaxis) + o
            zaxis1 = np.dot(r, zaxis1) + o
            xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
            xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
            xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
            yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
            p1 = (int(xp1), int(yp1))
            xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
            cv2.circle(frame, p2, 3, (255, 0, 0), 2)
            return frame

        position_half_face = (frame.shape[1] / 2, frame.shape[0] / 2, 0)
        draw_axes(frame, position_half_face, hp_out[0], hp_out[1], hp_out[2], 50, 950)


    def draw_gaze(self, frame, gaze_vector, left_eye_img, right_eye_img, eye_coords, frame_num):
        # left gaze
        left_eye_src_x = eye_coords[0]
        left_eye_src_y = eye_coords[1]
        left_eye_dest_x = left_eye_src_x + int(gaze_vector[0] * 90)
        left_eye_dest_y = left_eye_src_y + int(gaze_vector[1] * 90 * -1)

        cv2.arrowedLine(frame, (left_eye_src_x, left_eye_src_y), (left_eye_dest_x, left_eye_dest_y), (0, 255, 0), 2)

        # right gaze
        right_eye_src_x = eye_coords[2]
        right_eye_src_y = eye_coords[3]
        right_eye_dest_x = right_eye_src_x + int(gaze_vector[0] * 90)
        right_eye_dest_y = right_eye_src_y + int(gaze_vector[1] * 90 * -1)

        cv2.arrowedLine(frame, (right_eye_src_x, right_eye_src_y), (right_eye_dest_x, right_eye_dest_y), (0, 255, 0), 2)

        # save the image
        cv2.imwrite(f"output_vis/gaze_frame_{frame_num}.jpg", frame)

