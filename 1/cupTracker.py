"""
Owen Kephart & Mayank Agrawal
Read in a video and use a Temporal Average Threshold in order
to isolate moving cups

"""

import cv2
import numpy as np
import sys
import cvk2

MINAREA = 256 # for contours

# Read in the video - based on Zucker's original code

input = None

if len(sys.argv) != 3:
    print "usage: python cupTracker.py [video] [numObjects]"
    print "e.g. python cupTracker.py cups.mov 2"
    sys.exit(1)

inputFilename = sys.argv[1]
filename = inputFilename[:inputFilename.find(".")] # remove extension
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
print "Pre-Processing (may take long depending on how much memory your system has)"

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
cv2.imwrite(filename + 'avgFrame.png', avgFrame)
print "Calculated Background"
# taking out ending frames due to hand ending video
npAllFrames = np.stack(allFrames[:-30])

# keep track of original
rgbAllFrames = npAllFrames.astype('uint8')

# find difference b/w each frame and average
# multiple lines for efficiency
npAllFrames -= avgFrame
npAllFrames /= 2
npAllFrames += 127
npAllFrames = npAllFrames.astype('uint8')
print "Found Background Difference"
# store frames after thresholding
allThresholds = []
for i, frame in enumerate(npAllFrames):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    
    # erode thin edges
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.erode(frame,kernel, iterations = 3)
    allThresholds.append(frame)

    # sample middle thresholded frame
    if i == len(npAllFrames)/2:
        cv2.imwrite(filename + 'thresholdFrame.png', frame)
    """
    # Display the output image and wait for a keypress.
    cv2.imshow('Highlighted', frame)
    k = cv2.waitKey(10)
    """
print "Done Thresholding"
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

    areas = [] # for centroids
    
    # For each contour in the image
    for j in range(len(contours)):

        curContour = contours[j]
        
        # find the contours larger than a minimum area
        if cv2.contourArea(curContour) > MINAREA:
            areas.append((cv2.contourArea(curContour), curContour))

    # only want to look at biggest contours
    areas.sort(key = lambda tup: tup[0]) 
    
    if len(areas) >= numObjects:

        for j in range(numObjects):

            curContour = areas[j][1]
            x, y, w, h = cv2.boundingRect(curContour)
            centroid = (x + w/2, y + h/2)
            
            # find which cup the centroid belongs to

            if len(allCentroids) >= numObjects:
                # find centroid in last frame that was closest
                norms = map(lambda pos: np.linalg.norm(np.subtract(pos,centroid)), \
                                        prevCentroids)
                cupNum = np.argmin(norms)
            else:
                cupNum = j

            # append this centroid to lists
            allCentroids.append((centroid, cupNum))
            curCentroids[cupNum] = centroid

            # draw rectangle around centroid
            color = cvk2.getccolors()[cupNum]
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            # update previous centroids and append to line endpoints
            if prevCentroids[cupNum] != None:
                lineEndpoints.append((prevCentroids[cupNum], curCentroids[cupNum], color))
                prevCentroids[cupNum] = (sys.maxint, sys.maxint)

        prevCentroids = curCentroids[:] # copy elements instead of point

    # draw all lines
    for p1, p2, color in lineEndpoints:
            cv2.line(display, p1, p2, color)

    # Display the output image and wait for a keypress.
    cv2.imshow('Highlighted', display)
    k = cv2.waitKey(10)


# plot lines on blank black background
plot = np.zeros_like(display)

# draw all lines
for p1, p2, color in lineEndpoints:
        cv2.line(plot, p1, p2, color)

cv2.imwrite(filename + 'plots.png', plot)


