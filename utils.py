""" 
Copyright (C) Cortic Technology Corp. - All Rights Reserved
Written by Michael Ng <michaelng@cortic.ca>, 2021
"""

import cv2
import numpy as np

FINGER_COLOR = [
    (128, 128, 128),
    (80, 190, 168),
    (234, 187, 105),
    (175, 119, 212),
    (81, 110, 221),
]

JOINT_COLOR = [(0, 0, 0), (125, 255, 79), (255, 102, 0), (181, 70, 255), (13, 63, 255)]

class circularlist(object):
    def __init__(self, size, data = []):
        """Initialization"""
        self.index = 0
        self.size = size
        self._data = list(data)[-size:]

    def append(self, value):
        """Append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        """Get element by index, relative to the current index"""
        if len(self._data) == self.size:
            return(self._data[(key + self.index) % self.size])
        else:
            return(self._data[key])

    def __repr__(self):
        """Return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'

    def calc_average(self):
        num_data = len(self._data)
        sum = 0
        if num_data == 0:
            return 0
        for val in self._data:
            sum = sum + val
        return(float(sum)/num_data)


def draw_object_imgs(image, object_img, x1, y1, x2, y2, alpha):
    if x1 >= 0 and y1 >= 0 and x2 < image.shape[1] and y2 < image.shape[0]:
        object_alpha = object_img[:, :, 3] / 255.0
        combined_alpha = object_alpha * alpha
        y2 = y2 + (object_img.shape[0] - (y2 - y1))
        image[y1:y2, x1:x2, 0] = (1.0 - combined_alpha) * image[
            y1:y2, x1:x2, 0
        ] + combined_alpha * object_img[:, :, 0]
        image[y1:y2, x1:x2, 1] = (1.0 - combined_alpha) * image[
            y1:y2, x1:x2, 1
        ] + combined_alpha * object_img[:, :, 1]
        image[y1:y2, x1:x2, 2] = (1.0 - combined_alpha) * image[
            y1:y2, x1:x2, 2
        ] + combined_alpha * object_img[:, :, 2]

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def draw_hand_landmarks(img, hands, zoom_mode, single_handed):
    list_connections = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    for hand in hands:
        lm_xy = []
        for landmark in hand.landmarks:
            lm_xy.append([int(landmark[0]), int(landmark[1])])
        palm_line = [np.array([lm_xy[point] for point in [0, 5, 9, 13, 17, 0]])]
        cv2.polylines(img, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(len(list_connections)):
            finger = list_connections[i]
            line = [np.array([lm_xy[point] for point in finger])]
            cv2.polylines(img, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
            for point in finger:
                pt = lm_xy[point]
                cv2.circle(img, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
        if single_handed:
            if zoom_mode:
                cv2.line(img, (lm_xy[4][0], lm_xy[4][1]), (lm_xy[8][0], lm_xy[8][1]), (0, 255, 0), 2, cv2.LINE_AA)
    return img


def draw_zoom_scale(frame, translation_z, max_scale, screen_height):
    scale = (translation_z + max_scale) / (max_scale - (-max_scale))
    bar_height = int(scale * (screen_height // 3 + 2))
    cv2.rectangle(frame, (40, screen_height - 40), (60, screen_height - 40 - screen_height // 3), (255, 255, 255), 1, 1)
    cv2.rectangle(frame, (40 + 2, screen_height - 40 - 1), (60 - 2, screen_height - 40 - 1 - bar_height), (80, 190, 168), -1, 1)