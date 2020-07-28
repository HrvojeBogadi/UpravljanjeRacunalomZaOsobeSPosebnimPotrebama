import functions as fn
import dlib
import cv2 as cv
import numpy as np

#TODO Convert while loop into a function
#TODO Enumerate x and y coordinates

capture = cv.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("bin\shape_predictor_68_face_landmarks.dat")

roiFound = False

roi_x1, roi_x2, roi_y1, roi_y2 = 0, 0, 0, 0

def determinGazePoint(coordinate, numberOfSamples): #Coordinate is represented by numbers (0 -> X, 1 -> Y)
    avg = 0
    i = numberOfSamples
    global roiFound
    global roi_x1, roi_x2, roi_y1, roi_y2

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
            if not roiFound:
                roi_x1, roi_x2, roi_y1, roi_y2 = fn.findROI(gray, landmarks)
                roiFound = True
            roi = gray[roi_y1 : roi_y2, roi_x1 : roi_x2]
            roi = fn.resizeImage(roi)
            standImg = fn.standardizeImageBrightness(roi)
            edge = fn.findIrisEdge(standImg)
            center = fn.fitEllipse(edge)

            if center[coordinate] is not 0:
                avg += center[coordinate]
            i -= 1
            cv.waitKey(1)
    avgCoordinate = int(avg/numberOfSamples)

    return avgCoordinate

def calibrate(screen_width, screen_height):
    i = 25
    avg = 0

    cv.namedWindow("fs", cv.WINDOW_NORMAL)
    cv.setWindowProperty("fs", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    calibration_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    cv.circle(calibration_image, (25, int(screen_height / 2)), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    x_min = determinGazePoint(0, 20)


    cv.circle(calibration_image, (screen_width - 25, int(screen_height / 2)), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    x_max = determinGazePoint(0, 20)
    

    cv.circle(calibration_image, (int(screen_width / 2), 25), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    y_min = determinGazePoint(1, 20)

    
    cv.circle(calibration_image, (int(screen_width / 2), screen_height - 25), 25, (100, 100, 2), 2)
    cv.imshow("fs", calibration_image)
    cv.waitKey(2000)
    y_max = determinGazePoint(1, 20)


    print (x_min, x_max, y_min, y_max)
    return x_min, x_max, y_min, y_max

    capture.release()
    cv.destroyAllWindows()