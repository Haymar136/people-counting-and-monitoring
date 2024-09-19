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
import config
import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData

os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "/path/to/kernel/config/cache"

# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', default='videos/vid1.mp4', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
#modelConfiguration = "model/yolov3.cfg"
#modelWeights = "model/yolov3.weights"
modelConfiguration = "model/yolov4-tiny.cfg"
modelWeights = "model/yolov4-tiny.weights"

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
date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

config.in_count3 = 0
config.out_count3 = 0
config.in_time3 = date_time
config.out_time3 = date_time

#API_ENDPOINT = config.API_ENDPOINT
#API_CAM1 = config.API_CAM1
connection_string = config.conn3
event_hub_name = config.event_hub3

# Process inputs
winName = 'Camera 3'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

async def send_json_message(connection_string, event_hub_name, json_message):
    # Create an Event Hub producer client
    producer_client = EventHubProducerClient.from_connection_string(
        conn_str=connection_string, eventhub_name=event_hub_name
    )

    async with producer_client:
        # Create a batch.
        event_data_batch = await producer_client.create_batch()

        # Add events to the batch.
        event_data_batch.add(EventData(json.dumps(json_message)))
       
        # Send the batch of events to the event hub.
        await producer_client.send_batch(event_data_batch)

    # Close the producer client
    await producer_client.close()

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
            
            json_message = {
                    'store_name': 'Store 3',
                    'in_count': config.in_count3,
                    'out_count': config.out_count3,
                    'in_time': config.in_time3,
                    'out_time': config.out_time3
            } 
            
            asyncio.run(send_json_message(connection_string, event_hub_name, json_message))             
                       
def log_data(move_in, in_time, move_out, out_time):
	# function to log the counting data
	data = [move_in, in_time, move_out, out_time]
	# transpose the data to align the columns properly
	export_data = zip_longest(*data, fillvalue = '')

	with open('C:/Users/asus/Nynox Pte Ltd/Nynox Team - Documents/File/logs/counting_data.csv', 'w', newline = '') as myfile:
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
                # People Out count                
                if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalUp += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

                    config.out_count3 += 1
                    config.out_time3 = date_time

                    move_out.append(totalUp)
                    out_time.append(date_time)
                    types = 'OUT'
                    to.counted = True
                    print("[INFO] totalUP...", totalUp)
                    print("[INFO] out_time...", out_time)
                    log_data(move_in, in_time, move_out, out_time)
                
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                # People IN count  
                elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalDown += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

                    config.in_count3 += 1
                    config.in_time3 = date_time

                    move_in.append(totalDown)
                    in_time.append(date_time)
                    types = 'IN'
                    to.counted = True
                    print("[INFO] totaldown...",totalDown)
                    print("[INFO] in_time...", in_time)
                    log_data(move_in, in_time, move_out, out_time)
    
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

if __name__ == "__main__":
    if args.video:
        if not os.path.isfile(args.video):
            print(f"Input video file {args.video} doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = f"{args.video[:-4]}_output.avi"
    else:
        cap = cv.VideoCapture(0)

    # Reducing frame rate
    frameRate = 5

    # Reducing frame size
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    resized_width = 640
    resized_height = 450

    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), frameRate, (resized_width, resized_height))

    while cv.waitKey(1) < 0:
        frameId = cap.get(1)
        hasFrame, frame = cap.read()

        # Resize frame
        frame = cv.resize(frame, (resized_width, resized_height))

        if hasFrame and frameId % (frameRate * 2) == 0:
            cv.line(frame, (0, resized_height // 2), (resized_width, resized_height // 2), (0, 255, 255), 2)

        if not hasFrame:
            print("Done processing !!!")
            print(f"Output file is stored as {outputFile}")
            cv.waitKey(5000)
            cap.release()
            break

        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        t, _ = net.getPerfProfile()
        label = f'Inference time: {t * 1000.0 / cv.getTickFrequency():.2f} ms'
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        vid_writer.write(frame.astype(np.uint8))
        cv.moveWindow(winName, 20, 560)
        cv.imshow(winName, frame)


# Usage:  python counting_people1.py --video test.mp4
