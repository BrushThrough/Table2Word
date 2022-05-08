import cv2
from scipy import misc, ndimage
import numpy as np
import math
import os

class Photo_init():
    def __init__(self):
        pass

    def __call__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.adaptiveThreshold(~self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
        return self.hough()

    def hough(self):
        #±ßÔµ¼ì²â
        self.edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        # »ô·ò±ä»»
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 0)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

        if x1 == x2 or y1 == y2:
            # print("yes")
            return self.binary, self.img
        t = float(y2 - y1) / (x2 - x1)

        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        self.rotate_img = ndimage.rotate(self.binary, rotate_angle)
        self.rotate_img2 = ndimage.rotate(self.img, rotate_angle)
        return self.rotate_img,self.rotate_img2