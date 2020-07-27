import dlib

face_detector = dlib.get_frontal_face_detector()

def findFace(gray):
    x1, y1, x2, y2 = 0, 0, 0, 0

    faces = face_detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

    return x1, y1, x2, y2