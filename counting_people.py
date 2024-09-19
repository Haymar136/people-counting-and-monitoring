import csv
from itertools import zip_longest
import json
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
import datetime
import os
 
# importing the requests library
import requests

os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "/path/to/kernel/config/cache"

# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', default='road_test3.mp4', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
#modelConfiguration = "model/yolov3.cfg"
#modelWeights = "model/yolov3.weights"
modelConfiguration = "yolov4-tiny.cfg"
modelWeights = "yolov4-tiny.weights"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# initialize the video writer
writer = None
 
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
 
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0
move_out = []
move_in =[]
out_time = []
in_time = []

# types = []

# api-endpoint
#URL = "http://maps.googleapis.com/maps/api/geocode/json"
# URL = "https://mes.moeedu.org:8443/api/institutions/alllist"
 
# # location given here
# location = "delhi technological university"
 
# # defining a params dict for the parameters to be sent to the API
# PARAMS = {'address':location}

# defining the api-endpoint
#API_ENDPOINT = "http://localhost:8080/api/updateTime"

API_ENDPOINT_IN = "https://api.powerbi.com/beta/933c9cbe-35d3-4416-abbd-ddd1bca5879c/datasets/614e3d4d-d791-4704-ac54-cd4822623d3e/rows?clientSideAuth=0&experience=power-bi&referrer=embed.appsource&key=m2iIFMRWCeIdlDXNnXgC81J1o2x36CbCRGYfpKt3MEavVaAJp2eG%2FXhK6c5WZI2tr1UdHvMWdtf9llsdmu0CLg%3D%3D"

API_ENDPOINT_OUT = "https://api.powerbi.com/beta/933c9cbe-35d3-4416-abbd-ddd1bca5879c/datasets/614e3d4d-d791-4704-ac54-cd4822623d3e/rows?clientSideAuth=0&experience=power-bi&referrer=embed.appsource&key=m2iIFMRWCeIdlDXNnXgC81J1o2x36CbCRGYfpKt3MEavVaAJp2eG%2FXhK6c5WZI2tr1UdHvMWdtf9llsdmu0CLg%3D%3D"

headers = {'content-type': 'application/json'}


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            counting(objects)


def log_data(move_in, in_time, move_out, out_time):
	# function to log the counting data
	data = [move_in, in_time, move_out, out_time]
	# transpose the data to align the columns properly
	export_data = zip_longest(*data, fillvalue = '')

	with open('utils/data/logs/counting_data.csv', 'w', newline = '') as myfile:
		wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
		if myfile.tell() == 0: # check if header rows are already existing
			wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
			wr.writerows(export_data)

def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
 
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
 
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            #print(direction)
            to.centroids.append(centroid)
 
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalUp += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    move_out.append(totalUp)
                    out_time.append(date_time)
                    types = 'OUT'
                    to.counted = True
                    print("[INFO] totalUP...", totalUp)
                    print("[INFO] out_time...", out_time)
                    log_data(move_in, in_time, move_out, out_time)
                    
                    # sending get request and saving the response as response object
                    #r = requests.get(url = URL, params = PARAMS)
                    #r = requests.get(url = URL)
                    
                    # extracting data in json format
                    #data = r.json()
                    #print('data',data)
                    
                    # extracting latitude, longitude and formatted address
                    # of the first matching location
                    # latitude = data['results'][0]['geometry']['location']['lat']
                    # longitude = data['results'][0]['geometry']['location']['lng']
                    # formatted_address = data['results'][0]['formatted_address']
 
                    # # printing the output
                    # print("Latitude:%s\nLongitude:%s\nFormatted Address:%s"%(latitude, longitude,formatted_address)) 
                    
                    # data to be sent to api
                    body1 = {
                            'type': types,
                            'user_id': totalUp,
                            'time': date_time}
                    
                    print(body1)
 
                    # sending post request and saving response as response object
                    requests.post(url=API_ENDPOINT_OUT, data = json.dumps(body1), headers=headers)      
                    
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalDown += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    move_in.append(totalDown)
                    in_time.append(date_time)
                    types = 'IN'
                    to.counted = True
                    print("[INFO] totaldown...",totalDown)
                    print("[INFO] in_time...", in_time)
                    log_data(move_in, in_time, move_out, out_time)
                    
                    # sending get request and saving the response as response object
                    #r = requests.get(url = URL, params = PARAMS)
                    
                    # extracting data in json format
                    #data = r.json()
                    #print("data", data)
                    # extracting latitude, longitude and formatted address
                    # of the first matching location
                    # latitude = data['results'][0]['geometry']['location']['lat']
                    # longitude = data['results'][0]['geometry']['location']['lng']
                    # formatted_address = data['results'][0]['formatted_address']
 
                    # # printing the output
                    # print("Latitude:%s\nLongitude:%s\nFormatted Address:%s"%(latitude, longitude,formatted_address)) 
                    
                     # data to be sent to api
                    body = {
                            'type': types,
                            'user_id': totalDown,
                            'time': date_time}
                    
                    print(body)
                    # sending post request and saving response as response object
                    requests.post(url=API_ENDPOINT_IN, data = json.dumps(body), headers=headers) 
 
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #text = "ID {}".format(objectID)
        #cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):

        text = "{}".format(v)
        if k == 'Up':
            cv.putText(frame, f'Up : {text}', (10, 55),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if k == 'Down':
            cv.putText(frame, f'Down : {text}', (10, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Process inputs
winName = 'People Counting and Tracking System'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_output.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 10, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
frameRate = cap.get(3)

while cv.waitKey(1) < 0:
    
    # get frame from the video
    frameId = cap.get(1) #current frame number
    hasFrame, frame = cap.read()
    if hasFrame:
        if (frameId % math.floor(frameRate) == 0):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

    vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)


# Usage:  python3 counting_people.py --video D:/pycharmprojects/Counting-People/test.mp4
