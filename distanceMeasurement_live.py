import cv2
from objectDetectionModule import objectDetector


detector = objectDetector()

cap = cv2.VideoCapture('videos/test_video.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))/2

out = cv2.VideoWriter('output/output3.avi', cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))
while True:
    ret, frame = cap.read()

    if not ret:
        break

    detect_img = detector.object_detect(frame)

    out.write(detect_img)
    # cv2.imshow("Object Detection", detect_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

