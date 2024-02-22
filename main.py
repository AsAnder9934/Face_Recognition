import cv2
import numpy as np
import face_recognition

#wzór
malinowski = face_recognition.load_image_file('photos/image_MALINOWSKI.png')
malinowski = cv2.cvtColor(malinowski, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(malinowski)[0]
encode = face_recognition.face_encodings(malinowski)[0]

cv2.rectangle(malinowski, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

#kamera
video_capture = cv2.VideoCapture(0)

while True:
    ret, test = video_capture.read()
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

    face_loc_cam = face_recognition.face_locations(video_capture)[0]
    encode_cam = face_recognition.face_encodings(video_capture)[0]

    cv2.rectangle(video_capture, (face_loc_cam[3], face_loc_cam[0]), (face_loc_cam[1], face_loc_cam[2]), (255, 0, 255), 2)

    cv2.imshow('Okno', malinowski)
    cv2.imshow('Okno 2', test)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

video_capture.release()
cv2.destroyAllWindows()
