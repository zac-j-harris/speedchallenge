# 
# @author - zac_j_harris
# 

import numpy as np
import cv2
from statistics import median
import math
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MainLogger")
logger.setLevel(logging.DEBUG)

# logging.basicConfig(level=logging.DEBUG)


def dist(lst, out):
    for i in range(len(lst)):
        pt = lst[i]
        out[i] = (math.sqrt(pt[0]**2 + pt[1]**2))
        if out[i] > 15:
            out[i] = 0.0
    # print(out)
    return out


cap = cv2.VideoCapture('./data/train.mp4')
SAMPLE_SCALE = 10  # in number of pixels viewed
VIDEO_DURATION = 100  # in frames
# VIDEO_DURATION = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

logger.debug("Scale = {0} \tVideo Duration = {1}".format(SAMPLE_SCALE, VIDEO_DURATION))


# Parameters for lucas kanade optical flow

lk_params = {
    "winSize": (21,21),
    "maxLevel": 5,
    "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}



with open("./data/train.txt", "r+") as file:
    txt = file.read()

speed = [float(i) for i in txt.split("\n")]


# Get first frame
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Make initial sample points vector
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width*height % 2 == 0:
    p0 = np.empty([width*height//SAMPLE_SCALE**2,2],dtype=np.float32)
else:
    p0 = np.empty([width*height//SAMPLE_SCALE**2 + 1,2],dtype=np.float32)

# Make velocity vector
frame_vel = np.empty([p0.shape[0], 1])

for x in range(0,width,SAMPLE_SCALE):
    for y in range(0,height,SAMPLE_SCALE):
        p0[(x*height+y)//SAMPLE_SCALE**2] = [y*1.0, x*1.0]




frame_avg_flow = np.empty([VIDEO_DURATION, 1])
frame_med_flow = np.empty([VIDEO_DURATION, 1])
frame_transform = np.empty([VIDEO_DURATION, 1])
rel_frame_count = 0
abs_frame_count = 1

logger.debug("Values initialized")

while rel_frame_count < VIDEO_DURATION:
    ret, newframe = cap.read()
    abs_frame_count += 1

    if newframe is None:
        logger.info('video is completed')
        break

    new_gray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, new_gray, p0, None, **lk_params)

    if p1 is None:
        prev_gray = new_gray.copy()
        continue

    frame_vel = dist(p1, frame_vel)
    frame_avg_flow[rel_frame_count] = (sum(frame_vel) / len(frame_vel))
    frame_med_flow[rel_frame_count] = (median(frame_vel.copy()))
    if frame_med_flow[rel_frame_count] == 0.0:
        frame_transform[rel_frame_count] = frame_transform[rel_frame_count-1]
    else:
        frame_transform[rel_frame_count] = speed[abs_frame_count] / frame_med_flow[rel_frame_count]
    prev_gray = new_gray.copy()
    p0 = p1

    rel_frame_count += 1

    # Uncomment to watch video
    # cv2.imshow('frame',newframe)
    # k = cv2.waitKey(50) & 0xff
    # if k == 2:
    #     break

avg_video_transform = sum(frame_transform) / rel_frame_count
logger.info("Avg transform: " + str(avg_video_transform))
pred_speed = [i * avg_video_transform for i in frame_med_flow]

# print(rel_frame_count)
avg_video_vel = sum(frame_avg_flow) / rel_frame_count
avg_med_video_vel = sum(frame_med_flow) / rel_frame_count

logger.info("Average vel: " + str(avg_video_vel))
logger.info("Median vel: " + str(avg_med_video_vel))
logger.info("MSE: " + str(np.square(np.subtract(speed[1:rel_frame_count+1],pred_speed)).mean()))

cv2.destroyAllWindows()
cap.release()


# FULL VIDEO VALUES @ 1/50^2 pixels sampled:
#
# Avg transform:  [4.98921887]
# Average vel:  [0.00096308]
# Median vel:  [0.00107201]
# MSE:  215.72441407474403

