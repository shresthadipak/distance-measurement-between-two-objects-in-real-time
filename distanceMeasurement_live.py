import cv2
from objectDetectionModule import objectDetector

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    detector = objectDetector()

    detect_img = detector.object_detect(frame)

    cv2.imshow("Object Detection", detect_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

