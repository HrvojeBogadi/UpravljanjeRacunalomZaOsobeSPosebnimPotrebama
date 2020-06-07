import cv2 as cv
import numpy as np
#import target_vector_calculation as tvc

#   TODO: create a way of blink tracking (check if haarcascade
#       can be used)

cv.namedWindow("Adjustment Menu") #Mention CV and basic purpose

def on_change(val):
    pass   
cv.createTrackbar("Threshold Adjustment", "Adjustment Menu", 75, 255, on_change)


cap = cv.VideoCapture("data/testvid/di.mp4")

def findPupilCentre(feed):
    feed = cv.cvtColor(feed, cv.COLOR_BGR2HSV)
    contrast = 1.9
    brightness = -20
    feed[:,:,2] = np.clip(contrast * feed[:,:,2] + brightness, 0, 255)
    feed = cv.cvtColor(feed, cv.COLOR_HSV2BGR)

    gray = cv.cvtColor(feed, cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray, 5, 75, 75)
    cv.imshow("gray img", blur)
    threshVal = cv.getTrackbarPos("Threshold Adjustment", "Adjustment Menu")
    thresh = cv.threshold(blur, threshVal, 255, cv.THRESH_BINARY_INV)[1]
    canny = cv.Canny(thresh, 20, 200)
    contours = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(2,2))[0]
    cv.imshow("Adjustment Menu", thresh)

    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse = False)

    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    #Check if radius has values - in case an outline of pupil is not found
    if radius:
        j = radius.index(max(radius))
    else:
        return 0, 0

    x = int(centers[j][0])
    y = int(centers[j][1])

    cv.imshow("Adjust Thresh", thresh)

    return x,y


def findIredCentre(feed):
    orig = feed
    feed = cv.cvtColor(feed, cv.COLOR_BGR2HSV)
    contrast = 1.99
    brightness = -350
    feed[:,:,2] = np.clip(contrast * feed[:,:,2] + brightness, 0, 255)
    feed = cv.cvtColor(feed, cv.COLOR_HSV2BGR)

    cv.normalize(feed, feed, 0, 255, cv.NORM_MINMAX)

    gray = cv.cvtColor(feed, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7,7), 0)
    thresh = cv.threshold(blur, 25, 255, cv.THRESH_BINARY_INV)[1]
    canny = cv.Canny(thresh, 20, 200)
    contours = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(2,2))[0]

    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse = False)

    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    #Check if radius has values - in case an outline of pupil is not found
    if radius:
        j = radius.index(max(radius))
    else:
        return 0, 0

    x = int(centers[j][0])
    y = int(centers[j][1])

    return x,y

while(True):
    ret, frame = cap.read()
    if ret is False:break
    roi = frame[0:300, 0:500]
    rows, cols,_ = roi.shape

    cv.namedWindow("Adjustment Menu")
    
    px, py = findPupilCentre(roi)

    lx, ly = findIredCentre(roi)

    cv.line(roi, (px, py), (lx, ly), (0,255,0), 1)

    cv.imshow("Adjustment Menu", roi)
    # Display the resulting frame
    if cv.waitKey(100) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
