# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:29:08 2022

@author: Fahim H Khan (fkhan4@ucsc.edu)

UC Santa Cruz Noncommercial License

Acceptance
In order to get any license under these terms, you must agree to them as both strict obligations and conditions to all your licenses.

Copyright License
The licensor grants you a copyright license for the software to do everything you might do with the software that would otherwise infringe the licensor's copyright in it for any permitted purpose. However, you may only distribute the software according to Distribution License and make changes or new works based on the software according to Changes and New Works License.

Distribution License
The licensor grants you an additional copyright license to distribute copies of the software. Your license to distribute covers distributing the software with changes and new works permitted by Changes and New Works License.

Notices
You must ensure that anyone who gets a copy of any part of the software from you also gets a copy of these terms, as well as the following copyright notice:

This software is Copyright ©2022. The Regents of the University of California (“Regents”). All Rights Reserved.

Changes and New Works License
The licensor grants you an additional copyright license to make changes and new works based on the software for any permitted purpose.

Noncommercial Purposes
Any noncommercial purpose is a permitted purpose.

Commercial Purposes
Contact Innovation Transfer, UC Santa Cruz, innovation@ucsc.edu , https://officeofresearch.ucsc.edu/iatc/ , for any commercial purpose.

Personal Uses
Personal use for research, experiment, and testing for the benefit of public knowledge, personal study, private entertainment, hobby projects, amateur pursuits, or religious observance, without any anticipated commercial application, is use for a permitted purpose.

Noncommercial Organizations
Use by any charitable organization, educational institution, public research organization, public safety or health organization, environmental protection organization, or government institution is use for a permitted purpose regardless of the source of funding or obligations resulting from the funding.

Fair Use
You may have "fair use" rights for the software under the law. These terms do not limit them.

No Other Rights
These terms do not allow you to sublicense or transfer any of your licenses to anyone else, or prevent the licensor from granting licenses to anyone else.  These terms do not imply any other licenses.

Patent Defense
If you make any written claim that the software infringes or contributes to infringement of any patent, all your licenses for the software granted under these terms end immediately. If your company makes such a claim, all your licenses end immediately for work on behalf of your company.

Violations
The first time you are notified in writing that you have violated any of these terms, or done anything with the software not covered by your licenses, your licenses can nonetheless continue if you come into full compliance with these terms, and take practical steps to correct past violations, within 32 days of receiving notice.  Otherwise, all your licenses end immediately.

No Liability
As far as the law allows, the software comes as is, without any warranty or condition, and the licensor will not be liable to you for any damages arising out of these terms or the use or nature of the software, under any kind of legal claim.

Definitions
The "licensor" is Regents, and the "software" is the software the licensor makes available under these terms.
"You" refers to the individual or entity agreeing to these terms.
"Your company" is any legal entity, sole proprietorship, or other kind of organization that you work for, plus all organizations that have control over, are under the control of, or are under common control with that organization.  
"Control" means ownership of substantially all the assets of an entity, or the power to direct its management and policies by vote, contract, or otherwise.  Control can be direct or indirect.
"Your licenses" are all the licenses granted to you for the software under these terms.
"Use" means anything you do with the software requiring one of your licenses.

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
        cap = cv2.VideoCapture("https://stage-ams.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8")
        #cap = cv2.VideoCapture("")
        # ts = time.time()
        frameid = 0
        ret, frame = cap.read()
        # dest = str(ts) + ".avi"
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(dest, fourcc, 20.0, (1280, 720))

        while (cap.isOpened()):
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

        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    except:
        pass

