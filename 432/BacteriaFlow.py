import numpy as np
import cv2
import math


## There videos are provided. All the videos were recoded at VMI Biology Department Lab
cap = cv2.VideoCapture('paramecium2.flv')
# cap = cv2.VideoCapture('sphere.gif')
# params for ShiTomasi corner detection

feature_params = {
            "maxCorners":20,
            "qualityLevel":0.2,
            "minDistance": 20,
            "blockSize":21
            }
# Parameters for lucas kanade optical flow

lk_params = {
    "winSize": (21,21),
    "maxLevel": 8,
    "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.03)
}

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


## ShiTomasi corner detection algorithm
p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
distance_travel_full_video = 0

mask = np.zeros_like(frame)
while True:
    ret,newframe = cap.read()
    if newframe is None:
        print('video is completed')
        break
    frame_gray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is None:
        gray = frame_gray.copy()
        continue
    opticalflowpoints = p1[st==1]
    cornerpoints = p0[st==1]
    # draw the tracks
    distance_sum = 0
    for i,(opticalflow,corner) in enumerate(zip(opticalflowpoints,cornerpoints)):
        a,b = opticalflow.ravel()
        c,d = corner.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        newframe = cv2.circle(newframe,(a,b),5,color[i].tolist())
        individual_distance = math.sqrt((c-a)**2 + (d-b)**2)  # euclidian distance for each bacteria
        distance_sum += individual_distance  # total distance for 30 bacteria
    img = cv2.add(newframe,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(50) & 0xff
    if k == 2:
        break
    # Now update the previous frame and previous points
    # old_gray = frame_gray.copy()
    gray = frame_gray.copy()
    p0 = opticalflowpoints.reshape(-1,1,2)

    ## Now it is time to calcualte the velocity of the motion
    average_distance = distance_sum / max(cornerpoints.shape[0], 1.0)  # avg distance per bacteria
    distance_travel_full_video += average_distance
    ## Please fill blanks to calcualte velocity Unit: um/second
    ## Hint, you can count how many piexels the bacterial moved using the counter
    ## Meanwhile get the time passed during the motion
    ## Then based on distance/time, you can calculate the velocity


fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print("Velocity: ", distance_travel_full_video/duration)

# explain code, method, results for each piece of code



cv2.destroyAllWindows()
cap.release()
