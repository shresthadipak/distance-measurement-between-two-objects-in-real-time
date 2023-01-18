import cv2
import numpy as np
import matplotlib.pyplot as plt

yolov3_weights = "YOLOv3_model/yolov3.weights"
yolov3_cfg = "YOLOv3_model/yolov3.cfg"
coco_names = "YOLOv3_model/coco.names"

class objectDetector():

    def __init__(self, yolov3_weights = yolov3_weights, yolov3_cfg = yolov3_cfg, coco_names = coco_names):
        self.yolov3_weights = yolov3_weights
        self.yolov3_cfg = yolov3_cfg
        self.coco_names = coco_names

        self.yolo = cv2.dnn.readNet(self.yolov3_weights, self.yolov3_cfg)
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.yolo.getUnconnectedOutLayers()]

        with open(self.coco_names, "r") as file:
            self.classes = [line.strip() for line in file.readlines()] 

        self.colorRed = (0,0,255)
        self.colorGreen = (0,255,0)    

    def object_detect(self, img, draw=True):
        height, width, channels = img.shape

        # detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i, conf in zip(range(len(boxes)), confidences):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                text = label+ ' ' +str(round(conf, 4))

                if draw:
                    cv2.rectangle(img, (x, y), (x + w, y + h), self.colorGreen, 2)
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, self.colorRed, 2)
   
        return img
