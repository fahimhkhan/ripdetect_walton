# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:29:08 2022

@author: Fahim
"""

# use "pip install opencv-python" for installation
import cv2
import numpy as np
# use "pip install ultralytics" for installation
from ultralytics import YOLO
#from ultralytics import RTDETR
from datetime import datetime
import time

# Load yolov8 model
model = YOLO('yolov8x.pt')
#model = RTDETR('rtdetr-l.pt')

while True:
    try:
        #cap = cv2.VideoCapture("https://stage-ams.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")
        cap = cv2.VideoCapture("http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")
        #cap = cv2.VideoCapture("")
        # ts = time.time()
        frameid = 0
        ret, frame = cap.read()
        # dest = str(ts) + ".avi"
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(dest, fourcc, 20.0, (1280, 720))

        while (cap.isOpened()):
            now = datetime.now()
            current_time = now.strftime("%H")
            if int(current_time) > 7 and int(current_time) < 19:
                ret, frame = cap.read()
                if ret == True:
                    #img_boxes = cv2.resize(frame, (1280, 720))
                    img_boxes = frame
                    h, w, _ = frame.shape
                    #mask = np.zeros(frame.shape, dtype=np.uint8)

                    # Define the region to be masked (in this case, a rectangle)
                    #x, y, w, h = 50, 50, 100, 100
                    #mask[h-500:h, w-500:w] = 255

                    # Apply the mask to the image
                    #masked_image = cv2.bitwise_or(frame, mask)

                    #use YOLOv8
                    #results = model.predict(img_boxes, conf = 0.2)
                    results = model.predict(img_boxes)
                    for result in results:
                        for score, cls, bbox in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):
                            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                            #h, w, _ = frame.shape
                                
                            y_min = int(max(1, y1))
                            x_min = int(max(1, x1))
                            y_max = int(min(h, y2))
                            x_max = int(min(w, x2))

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            if cls.item() == 0.0:
                                label = "Rip Current"+ ": " + ": {:.2f}%".format(score * 100)                 
                                img_boxes = cv2.rectangle(img_boxes,(x_min, y_max),(x_max, y_min),(0,0,255), 4)
                                cv2.putText(img_boxes, label, (x_min, y_max-10), font, 1, (0,0,255), 1, cv2.LINE_AA)
                                current_time = datetime.now()
                                result = str(current_time) + "," + "Rip Current" + "," + str(score * 100) + '\n'
                                print(result)
                                logfname = str(current_time)
                                logfname = logfname.split(" ")[0] + '.csv'
                                with open(logfname, 'a') as the_file:
                                #with open('detection_log.csv', 'a') as the_file:
                                    the_file.write(result)

                    
                    outp = cv2.resize(img_boxes, (1280, 720))
                    # out.write(outp)
                    frameid += 1
                    cv2.imshow('PreviewWindow', outp)

                    k = cv2.waitKey(10) & 0xff
                    if k == 27:
                        break
                else:
                    break
            else:
                print("STREAM OFF")
            

        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    except:
        pass

