import numpy as np
import cv2 as cv

capture = cv.VideoCapture(0)

while(True):
    ret, frame = capture.read()

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()