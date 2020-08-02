import numpy as np
import cv2 as cv
import dlib
import functions as fn
import calibration as calib
from tkinter import Tk
from pynput.mouse import Controller

mouse = Controller()

root = Tk()
capture = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("bin/shape_predictor_68_face_landmarks.dat")

roiFound = False

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

center_x = 0
center_y = 0

x_min, x_max, y_min, y_max = calib.calibrate(face_detector, predictor, screen_width, screen_height)


while(True):
    _, frame = capture.read()
    gray = fn.convertImageToGray(frame)
    gray = cv.medianBlur(gray, 5)

    faces = face_detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        if not roiFound:
            roi_x1, roi_x2, roi_y1, roi_y2 = fn.findROI(gray, landmarks)
            roiFound = True
        roi = gray[roi_y1 : roi_y2, roi_x1 : roi_x2]
        roi = fn.resizeImage(roi)

        standImg = fn.standardizeImage(roi)

        edge = fn.findIrisEdge(standImg)

        gazeCenter = fn.fitEllipse(edge)

        gazeX, gazeY = gazeCenter[0], gazeCenter[1]

    cursor_x = int( (gazeX - x_min) * screen_width / (x_max - x_min) )
    cursor_y = int( (gazeY - y_min) * screen_height / (y_max - y_min) )

    mouse.position = (cursor_x, cursor_y)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()