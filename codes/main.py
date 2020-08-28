import numpy as np
import cv2 as cv
import dlib
import functions as fn
from pynput.mouse import Controller

mouse = Controller()

capture = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("bin/shape_predictor_68_face_landmarks.dat")

inverted = True

eyelidHeightAvg = 0
n = 25


while(True):
    _, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        roi_x1, roi_x2, roi_y1, roi_y2 = fn.findROI(gray, landmarks)
        roi = gray[roi_y1 : roi_y2, roi_x1 : roi_x2]
        standImg = fn.standardizeImage(roi)
        edge = fn.findIrisEdge(standImg)

        gazeCenter = fn.fitEllipse(edge)
        gazeCenter = gazeCenter[0] + roi_x1, gazeCenter[1] + roi_y1

        midPointTop, midPointBottom, _, _ = fn.findEyeCenter(landmarks)
        if (n > 0):
            eyelidHeightAvg += midPointBottom[1] - midPointTop[1]
            n -= 1
        elif (n == 0):
            eyelidHeightAvg = eyelidHeightAvg / 25
            n = -1


        if (n < 0):
            w1, w2 = fn.splitEyeRegionsVertical(landmarks)
            
            if not inverted:
                if gazeCenter[0] < w1:
                    mouse.move(-20, 0)
                elif gazeCenter[0] > w2:
                    mouse.move(20, 0)
            else:
                if gazeCenter[0] < w1:
                    mouse.move(20, 0)
                elif gazeCenter[0] > w2:
                    mouse.move(-20, 0)
            
            movement = fn.moveCursorVertical(landmarks, eyelidHeightAvg)
            if(movement == 1):
                mouse.move(0, 15)
            elif(movement == -1):
                mouse.move(0, -15)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()