import cv2 as cv
import numpy as np

#def findRegionOfInterest(feed):
#    return roi

#def adjustFeedForPupil(roi):
#    return roiPupil

#def adjustFeedForIred(roi):
#    return roiIred

#def findPupilCentre(roiPupil):
#    return x,y

#def findIredCentre(roiIred):
#    return x,y

#def calculatePupilIredVector(px, py, ix, iy):
#   return vector

#def calibrateDevice():
#    return something

#def determineGazePosition():
#    return x,y

eye_cascade = cv.CascadeClassifier('data/haarcascades/eye.xml')

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if ret is False:break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    eye = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eye:
        cv.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
        cv.circle(frame, (ex+int(ew/2),ey+int(eh/2)), 20, (0,0,255), 2)

    cv.imshow("image", frame)

    if cv.waitKey(100) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

cap.release()
cv.destroyAllWindows()
