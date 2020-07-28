import numpy as np
import cv2 as cv
import dlib
import functions as fn
import win32api as win
import calibration as calib

capture = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("bin\shape_predictor_68_face_landmarks.dat")

roiFound = False

screen_width = win.GetSystemMetrics(0)
screen_height = win.GetSystemMetrics(1)

center_x = 0
center_y = 0

x_min, x_max, y_min, y_max = calib.calibrate(screen_width, screen_height)

t = 5
avgX = 0
avgY = 0

while(True):
    _, frame = capture.read()
    gray = fn.convertImageToGray(frame)

    faces = face_detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        if not roiFound:
            roi_x1, roi_x2, roi_y1, roi_y2 = fn.findROI(gray, landmarks)
            roiFound = True
        roi = gray[roi_y1 : roi_y2, roi_x1 : roi_x2]
        roi = fn.resizeImage(roi)
        cv.imshow("Roi", roi)

        standImg = fn.standardizeImageBrightness(roi)

        edge = fn.findIrisEdge(standImg)

        center = fn.fitEllipse(edge)

        avgX += center[0]
        avgY += center[1]

        cv.imshow("Gray", gray)

    avgX /= 5
    avgY /= 5

    cursor_x = int( (avgX - x_min) * screen_width / (x_max - x_min) )
    cursor_y = int( (center[1] - y_min) * screen_height / (y_max - y_min) )

    win.SetCursorPos((cursor_x, cursor_y))

    t = 5
    avgX = 0
    avgY = 0

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()


