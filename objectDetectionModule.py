import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

        self.colorWhite = (255, 255, 255)   

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

        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        pixels_ratio_array = []

        for i, conf in zip(range(len(boxes)), confidences):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i]
                if label == 'person' or label == 'dog':
                    text = label+ ' ' +str(round(conf, 2))
                    if draw:
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorWhite, 1)


                if label == 'person':
                    height_person = 60
                    width_p = w
                    height_p = int(h/2)

                    # print(w, h)

                    # # pull bbox coordinate points
                    x0 = x
                    y0 = y
                    x1 = x + w
                    y1 = y + h


                    person_x_min = int(x0)
                    person_x_max = int(x1)
                    person_y_min = int(y0)
                    person_y_max = int(y1)

                    pixels_to_inches = h/height_person
                    pixels_ratio_array.append(pixels_to_inches)

                if label == 'dog':
                    height_dog = 28
                    width_d = w
                    height_d = int(h/2)

                    pixels_to_inches = h/height_dog
                    pixels_ratio_array.append(pixels_to_inches)

                    # pull bbox coordinate points
                    x0 = x
                    y0 = y
                    x1 = x + w
                    y1 = y + h

                    dog_x_min = int(x0)
                    dog_x_max = int(x1)
                    dog_y_min = int(y0)
                    dog_y_max = int(y1)

        try:
            if width_d and width_p:
                pixel_to_inches = int(np.mean(pixels_ratio_array)) 

                x_calc_1 = abs(person_x_min - dog_x_min)
                x_calc_2 = abs(person_x_min - dog_x_max)
                x_calc_3 = abs(person_x_max - dog_x_max)
                x_calc_4 = abs(person_x_max - dog_x_min) 

                # print(str(x_calc_1))
                # print(str(x_calc_2))
                # print(str(x_calc_3))
                # print(str(x_calc_4))

                # print (math.sqrt((person_x_min - dog_x_max)**2 + (person_y_min - dog_y_max)**2))

                min_pixel_distance = min(x_calc_1, x_calc_2, x_calc_3, x_calc_4)
                # print(min_pixel_distance)
                # print(pixel_to_inches)
                # print(pixels_ratio_array)

                if x_calc_1 == min_pixel_distance:
                        
                        estimated_distance_inches = round(min_pixel_distance / pixel_to_inches, 2)
                        distance_start_point = (dog_x_min, dog_y_max - height_d)
                        distance_end_point = (person_x_min, person_y_max - height_p)
                        distance_text_point = (dog_x_min + int(x_calc_1 / 5), (person_y_max - height_p) - 10)
                        
                        print("x_calc_1 - smallest: " + str(x_calc_1))
                        print("Estimated Inches: " + str(estimated_distance_inches))

                elif x_calc_2 == min_pixel_distance:
                        
                        estimated_distance_inches = round(min_pixel_distance / pixel_to_inches, 2)
                        distance_start_point = (dog_x_max, dog_y_max - height_d)
                        distance_end_point = (person_x_min, person_y_max - height_p)
                        distance_text_point = (dog_x_max + int(x_calc_2 / 5), (person_y_max - height_p) - 20)
                        
                        print("x_calc_2 - smallest: " + str(x_calc_2))
                        print("Estimated Inches: " + str(estimated_distance_inches))  

                elif x_calc_3 == min_pixel_distance:
                        
                        estimated_distance_inches = round(min_pixel_distance / pixel_to_inches, 2)
                        distance_start_point = (person_x_max, person_y_max - height_p)
                        distance_end_point = (dog_x_max, dog_y_max - height_d)
                        distance_text_point = (person_x_max + int(x_calc_3 / 5), (person_y_max - height_p) - 20)
                        
                        print("x_calc_3 - smallest: " + str(x_calc_3))
                        print("Estimated Inches: " + str(estimated_distance_inches)) 

                else:
                        
                        estimated_distance_inches = round(min_pixel_distance / pixel_to_inches, 2)
                        distance_start_point = (person_x_max, person_y_max - height_p)
                        distance_end_point = (dog_x_min, dog_y_max - height_d)
                        distance_text_point = (person_x_max + int(x_calc_4 / 5), (person_y_max - height_p) - 20)
                        
                        print("x_calc_4 - smallest: " + str(x_calc_4))
                        print("Estimated Inches: " + str(estimated_distance_inches))       

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (255, 255, 255)
                font_thickness = 2 
                font_scale = 1

                distance_color = (255, 255, 255)
                distance_thickness = 2
                if draw:
                    cv2.circle(img, distance_start_point, 5, (0, 0, 255), -1)
                    cv2.circle(img, distance_end_point, 5, (0, 0, 255), -1)
                    cv2.line(img, distance_start_point, distance_end_point, distance_color, distance_thickness)

                    cv2.putText(img, "~" + str(estimated_distance_inches) + " in.", distance_text_point, font, font_scale, font_color, font_thickness, cv2.LINE_AA)    

                return img
        except:
            pass
