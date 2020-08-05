import cv2

trained_face_data = cv2.CascadeClassifier("C:\\Users\\Sriz\\Desktop\\Face Ditection\\data\\haarcascade_frontalface_default.xml")

img = cv2.imread("tom_hardy.jpg")

gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

cv2.imshow("Face detector", img)
cv2.waitKey()
