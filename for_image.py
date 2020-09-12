import cv2
import os


trained_face_data = cv2.CascadeClassifier(".\\training_data\\haarcascade_frontalface_default.xml")

files = os.listdir(".\\input_data\\")

for file in files:
    img = cv2.imread(f".\\input_data\\{str(file)}")
    gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imwrite(f".\\output\\{str(file)}", img)
    print(file)
    cv2.waitKey(0)
