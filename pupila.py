import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)

    _, threshold = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 1:
            cv2.circle(eye_region, center, radius, (0, 255, 0), 2)

    return eye_region

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            gray_face = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(gray_face)
            eyes = sorted(eyes, key=lambda e: e[0])

            if len(eyes) >= 2:
                ex, ey, ew, eh = eyes[1]
                eye_region = face_roi[ey:ey+eh, ex:ex+ew]
                processed_eye = detect_pupil(eye_region)
                face_roi[ey:ey+eh, ex:ex+ew] = processed_eye

        cv2.imshow("Right Eye Pupil Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()