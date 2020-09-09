import cv2


trained_face_data = cv2.CascadeClassifier(".\\training_data\\haarcascade_frontalface_default.xml")

web_cam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = web_cam.read()
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow("Face detector", frame)

    cv2.waitKey(1)

