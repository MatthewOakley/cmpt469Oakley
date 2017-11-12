# practice python viola and jones
import os
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def runFaces(fileName):
  print(fileName)
  cv2.namedWindow("output", cv2.WINDOW_NORMAL)
  img = cv2.imread(fileName)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

  cv2.imshow('output', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
'''
for i in range(100):
  if(i >= 10):
    if(os.path.isfile("../Training/1-0" + str(i) + ".jpg")):
      runFaces( ("../Training/1-0" + str(i) + ".jpg") )
    else:
      runFaces( ("../Training/0-0" + str(i) + ".jpg") )

  else:
    if(os.path.isfile("../Training/1-00" + str(i) + ".jpg")):
      runFaces( ("../Training/1-00" + str(i) + ".jpg") )
    else:
      runFaces( ("../Training/0-00" + str(i) + ".jpg") )
'''
for i in range(100):
  if(i >= 10):
    if(os.path.isfile("../Testing/1-00" + str(i) + ".jpg")):
      runFaces( ("../Testing/1-00" + str(i) + ".jpg") )
    else:
      runFaces( ("../Testing/0-00" + str(i) + ".jpg") )

  else:
    if(os.path.isfile("../Testing/1-000" + str(i) + ".jpg")):
      runFaces( ("../Testing/1-000" + str(i) + ".jpg") )
    else:
      runFaces( ("../Testing/0-000" + str(i) + ".jpg") )