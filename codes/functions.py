import cv2 as cv
import numpy as np

left_eye_l = 36
left_eye_r = 39
left_eye_lt = 37
left_eye_rt = 38
left_eye_lb = 40
left_eye_rb = 41
right_eye_l = 42
right_eye_r = 45
right_eye_lt = 43
right_eye_rt = 44
right_eye_lb = 47
right_eye_rb = 46

offset = 10

resize_scale = 1

target_brightness = 0.50

kernel = np.ones((5,5),np.uint8)


def convertImageToGray(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    return gray

def resizeImage(frame):
    width = int(frame.shape[1] * resize_scale)
    height = int(frame.shape[0] * resize_scale)
    dimension = (width, height)

    return cv.resize(frame, dimension, interpolation = cv.INTER_AREA)

def findROI(frame, landmarks):
    roi_x1 = landmarks.part(left_eye_l).x-offset
    roi_x2 = landmarks.part(left_eye_r).x+offset
    roi_y1 = landmarks.part(left_eye_lt).y-offset
    roi_y2 = landmarks.part(left_eye_lb).y+offset

    roi = frame[roi_y1 : roi_y2, roi_x1 : roi_x2]

    roi = resizeImage(roi)

    return roi

def standardizeImageBrightness(frame):
    cols, rows = frame.shape
    brightness = np.sum(frame) / (255 * cols * rows)
    brightness_ratio = brightness / target_brightness
    frame = cv.convertScaleAbs(frame, alpha=(1 / brightness_ratio), beta=0)
    return frame

def findIrisEdge(frame):
    _, thresh = cv.threshold(frame, 64, 255, cv.THRESH_BINARY_INV)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    canny = cv.Canny(closing, 100, 300)
    return canny

def fitEllipse(edge):
    maxi = 0
    maxArea = 0
    center = (0, 0)

    contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ellipse = [None] * len(contours)

    for i, c in enumerate(contours):
        if c.shape[0] > 5:
            (x, y), (MA, ma), angle = cv.fitEllipse(c)
            ellipse[i] = ((x, y), (MA, ma), angle)
            if MA * ma * np.pi > maxArea:
                maxArea = MA * ma * np.pi
                maxi = i
                center = (int(x), int(y))
    
    if center is None:
        return
    return center