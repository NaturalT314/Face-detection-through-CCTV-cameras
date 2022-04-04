#!/usr/bin/python3
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

obama_alone_path = '/Users/ahmad/Downloads/obama_alone.jpeg'
obama_crowd_path = '/Users/ahmad/Downloads/obama.jpeg'

model_name = "VGG-Face"
model = DeepFace.build_model(model_name) 

face_cascade_default = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

obama_alone = cv2.imread(obama_alone_path)

gray = cv2.cvtColor(obama_alone, cv2.COLOR_BGR2GRAY)

faces = face_cascade_default.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

for (x, y, w, h) in faces:
	cv2.imwrite("./test.jpeg", obama_alone[y:y+h, x:x+w])
	cv2.rectangle(obama_alone, (x, y), (w+x, h+y), (0,0,255), thickness=5)

cv2.imshow('Obama Alone', obama_alone)


cv2.waitKey(0)

# plt.imshow(obama_alone[:, :, ::-1 ])
# plt.show()

# plt.imshow(obama_crowd[:, :, ::-1 ])
# plt.show()






# result = DeepFace.verify(obama_alone_path,obama_crowd_path)

# print(result)

# DeepFace.verify(obama_alone_path, obama_crowd_path)
