"""
Owen Kephart & Mayank Agrawal
Read in a video and use a Temporal Average Threshold in order
to isolate moving objects

"""

import cv2
import numpy as np
import sys
import cvk2


# Read in the video - based on Zucker's original code

input = None

if len(sys.argv) != 3:
    print "usage: python cupTracker.py [video] [numObjects]"
    sys.exit(1)

inputFilename = sys.argv[1]
numObjects = int(sys.argv[2])
try:
    input = int(inputFilename)
except:
    pass

capture = cv2.VideoCapture(inputFilename)
if capture:
    print 'Opened file', inputFilename

# Bail if error.
if not capture or not capture.isOpened():
    print 'Error opening video capture!'
    sys.exit(1)

# Fetch the first frame and bail if none.
ok, frame = capture.read()

if not ok or frame is None:
    print 'No frames in video'
    sys.exit(1)


# Finding Temporal Average

allFrames = []
allFrames.append(np.array(frame, dtype = 'int32'))
# Loop until movie is ended and add frames:
while True:
    # Get the frame.
    ok, frame = capture.read(frame)

    # Bail if none.
    if not ok or frame is None:
        break

    allFrames.append(np.array(frame, dtype = 'int32'))


# find background frame
npBackgroundFrames = np.stack(allFrames[:10])
avgFrame = np.median(npBackgroundFrames, axis = 0).astype('int32')

# taking out ending frames due to hand ending video
npAllFrames = np.stack(allFrames[:-30])

# keep track of original
rgbAllFrames = npAllFrames.astype('uint8')

print "calculating differences"
# find difference b/w each frame and average
# multiple lines for efficiency
npAllFrames -= avgFrame
npAllFrames /= 2
npAllFrames += 127
npAllFrames = npAllFrames.astype('uint8')
print "differences done"

# store frames after thresholding
allThresholds = []
for i, frame in enumerate(npAllFrames):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    
    # erode thin edges
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.erode(frame,kernel, iterations = 3)
    allThresholds.append(frame)

    if i%100 == 0:

        cv2.imwrite('thresholdFrame' + str(i) + '.png', frame)


    # Display the output image and wait for a keypress.
    cv2.imshow('Highlighted', frame)
    k = cv2.waitKey(10)

print "thresholded"
# store centroids of each cup
allCentroids = []
curCentroids = [None]*numObjects
prevCentroids = [None]*numObjects
lineEndpoints = []

# Get the list of contours in the image
for i, frame in enumerate(allThresholds):
    
    frame, contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)
    
    display = rgbAllFrames[i] #output frame

    areas = []
    
    # For each contour in the image
    for j in range(len(contours)):

        curContour = contours[j]
        areas.append((cv2.contourArea(curContour), curContour))

    areas.sort(key = lambda tup: tup[0])
    
    if len(areas) >= numObjects:

        for j in range(numObjects):

            curContour = areas[j][1]
            x, y, w, h = cv2.boundingRect(curContour)
            centroid = (x + w/2, y + h/2)
            
            if len(allCentroids) >= numObjects:
                # find centroid in last frame that was closest
                norms = map(lambda pos: np.linalg.norm(np.subtract(pos,centroid)), \
                                        prevCentroids)
                cupNum = np.argmin(norms)
            else:
                cupNum = j

            allCentroids.append((centroid, cupNum))
            curCentroids[cupNum] = centroid
            color = cvk2.getccolors()[cupNum]
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        if prevCentroids[0] != None:
            for j in range(numObjects):
                lineEndpoints.append((prevCentroids[j], curCentroids[j]))

        prevCentroids = curCentroids[:]

    for j in range(0, len(lineEndpoints) - numObjects, numObjects):
        for k in range(numObjects):
            cupNum = k
            p1, p2 = lineEndpoints[j+k]

            color = cvk2.getccolors()[cupNum]
            cv2.line(display, p1, p2, color)

    # Display the output image and wait for a keypress.
    cv2.imshow('Highlighted', display)
    k = cv2.waitKey(10)



