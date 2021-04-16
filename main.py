
import cv2 # imports OPENCV library

thres = 0.45 # Threshold to detect object

# dot represent that you are using particular function from the respective lib
cap = cv2.VideoCapture(0)  # Using CV2 lib, VideoCapture -> Capture Video from Camera
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= [] # Empty variable declared
classFile = 'coco.names' # Import File
with open(classFile,'rt') as f: # Imported file has opened
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# Neural Network Begins here
net = cv2.dnn_DetectionModel(weightsPath,configPath) # Detection model initialized
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Loop
while True:
    success,img = cap.read()  # Image Matrix
    classIds, confs, bbox = net.detect(img,confThreshold=thres) #
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow("Abhinav",img)
    cv2.waitKey(1)