'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2


#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

def eye_aspect_ratio(eye):
    """ Calculate eye aspect ratio """
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

def extract_eye(gray, face):
    """ Get array of coordinates of leftEye and rightEye """
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    return leftEye, rightEye

def get_eye_ratio(leftEye, rightEye):
    """ Calculate aspect ratio of both eyes """
    leftEyeAspectRatio = eye_aspect_ratio(leftEye)
    rightEyeAspectRatio = eye_aspect_ratio(rightEye)

    eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
    return eyeAspectRatio

def detect_drowsiness(frame):
    """ Return annotated image and drowsiness detection result """
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:
        leftEye, rightEye = extract_eye(gray, face)

        eyeAspectRatio = get_eye_ratio(leftEye, rightEye)

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            return {'image': frame, 'drowsy': True}
        else:
            return {'image': frame, 'drowsy': False}

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Give some time for camera to initialize(not required)
time.sleep(2)

if __name__ == '__main__':
    img = cv2.imread('images/drowsy_2.jpg')
    result = detect_drowsiness(img)
    img = result['image']
    print('drowsy: {}'.format(result['drowsy']))

    cv2.imshow('Video', img)

    while(True):
        #Read each frame and flip it, and convert to grayscale
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()