import functions as fn
import dlib
import cv2 as cv
import numpy as np

#TODO Convert while loop into a function

capture = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("bin\shape_predictor_68_face_landmarks.dat")

def calibrate(screen_width, screen_height):
    i = 25
    avg = 0

    cv.namedWindow("fs", cv.WINDOW_NORMAL)
    cv.setWindowProperty("fs", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    calibration_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    cv.circle(calibration_image, (25, int(screen_height / 2)), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    while(i > 0):
        _, frame = capture.read()
        gray = fn.convertImageToGray(frame)
        faces = face_detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)
            roi = fn.findROI(gray, landmarks)
            standImg = fn.standardizeImageBrightness(roi)
            edge = fn.findIrisEdge(standImg)
            center = fn.fitEllipse(edge)

            cv.imshow("E", edge)

            if center[0] is not 0:
                avg += center[0]
            i -= 1
            cv.waitKey(1)
    x_min = int(avg/25)
    i = 25
    avg = 0

    

    cv.circle(calibration_image, (int(screen_width / 2), 25), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    while(i > 0):
        _, frame = capture.read()
        gray = fn.convertImageToGray(frame)
        faces = face_detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)
            roi = fn.findROI(gray, landmarks)
            standImg = fn.standardizeImageBrightness(roi)
            edge = fn.findIrisEdge(standImg)
            center = fn.fitEllipse(edge)

            if center[1] is not 0:
                avg += center[1]
            i -= 1
            cv.waitKey(1)
    y_min = int(avg/25)
    i = 25
    avg = 0

    
    cv.circle(calibration_image, (int(screen_width / 2), screen_height - 25), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    while(i > 0):
        _, frame = capture.read()
        gray = fn.convertImageToGray(frame)
        faces = face_detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)
            roi = fn.findROI(gray, landmarks)
            standImg = fn.standardizeImageBrightness(roi)
            edge = fn.findIrisEdge(standImg)
            center = fn.fitEllipse(edge)

            if center[1] is not 0:
                avg += center[1]
            i -= 1
            cv.waitKey(1)
    y_max = int(avg/25)
    i = 25
    avg = 0

    
    cv.circle(calibration_image, (screen_width - 25, int(screen_height / 2)), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    while(i > 0):
        _, frame = capture.read()
        gray = fn.convertImageToGray(frame)
        faces = face_detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)
            roi = fn.findROI(gray, landmarks)
            standImg = fn.standardizeImageBrightness(roi)
            edge = fn.findIrisEdge(standImg)
            center = fn.fitEllipse(edge)

            if center[0] is not 0:
                avg += center[0]
            i -= 1
            cv.waitKey(1)
    x_max = int(avg/25)

    print (x_min, x_max, y_min, y_max)
    return x_min, x_max, y_min, y_max

    capture.release()
    cv.destroyAllWindows()