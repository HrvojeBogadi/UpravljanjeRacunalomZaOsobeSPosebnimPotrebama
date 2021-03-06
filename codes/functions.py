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

target_brightness = 0.6

kernel = np.ones((5,5),np.uint8)

def findROI(frame, landmarks):
    roi_x1 = landmarks.part(left_eye_l).x - offset
    roi_x2 = landmarks.part(left_eye_r).x + offset
    roi_y1 = landmarks.part(left_eye_lt).y - offset
    roi_y2 = landmarks.part(left_eye_lb).y + offset
    return roi_x1, roi_x2, roi_y1, roi_y2

def standardizeImage(frame):
    frame = cv.medianBlur(frame, 5)
    cols, rows = frame.shape
    brightness = np.sum(frame) / (255 * cols * rows)
    brightness_ratio = brightness / target_brightness
    frame = cv.convertScaleAbs(frame, alpha=(1 / brightness_ratio), beta=0)
    denoise = cv.fastNlMeansDenoising(frame, None, 15, 7, 21)
    return denoise

def findIrisEdge(frame):
    _, thresh = cv.threshold(frame, 70, 255, cv.THRESH_BINARY_INV)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    canny = cv.Canny(closing, 100, 300)
    return canny

def fitEllipse(edge):
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
                center = (int(x), int(y))
    
    if center is None:
        return
    return center

def findEyeCenter(landmarks):
    midPointTop = int( ( landmarks.part(left_eye_lt).x + landmarks.part(left_eye_rt).x ) / 2), int( ( landmarks.part(left_eye_lt).y + landmarks.part(left_eye_rt).y ) / 2)
    midPointBottom = int( ( landmarks.part(left_eye_lb).x + landmarks.part(left_eye_rb).x ) / 2), int( ( landmarks.part(left_eye_lb).y + landmarks.part(left_eye_rb).y ) / 2)
    midPointLeft = landmarks.part(left_eye_l).x, landmarks.part(left_eye_l).y
    midPointRight = landmarks.part(left_eye_r).x, landmarks.part(left_eye_r).y

    return midPointTop, midPointBottom, midPointLeft, midPointRight

def splitEyeRegionsVertical(landmarks):
    totalWidth = landmarks.part(left_eye_r).x - landmarks.part(left_eye_l).x
    areaWidth = int( totalWidth / 3 )

    w1 = landmarks.part(left_eye_l).x + areaWidth
    w2 = landmarks.part(left_eye_l).x + areaWidth * 2
    
    return w1, w2

def moveCursorVertical(landmarks, eyelidHeightAvg):
    midPointTop, midPointBottom, _, _ = findEyeCenter(landmarks)
    y1 = midPointBottom[1]
    y2 = midPointTop[1]
    direction = (y1 - y2) - eyelidHeightAvg
    offset = int( eyelidHeightAvg / 4 )

    if (direction > offset):
        return -1
    elif (direction < (0 - offset)):
        return 1