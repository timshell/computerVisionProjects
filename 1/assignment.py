"""
Owen Kephart & Mayank Agrawal
Read in a video and use a Temporal Average Threshold in order
to isolate moving objects

"""

import cv2
import numpy as np
import sys

thresholdVal = 50
avgFrame = None


def thresholdFrame(frameDiff):
    ret, thresh = cv2.threshold(frameDiff, thresholdVal, 255, cv2.THRESH_BINARY)
    return thresh


# Read in the video - Zucker's original code

# Figure out what input we should load:
input_device = None

if len(sys.argv) > 1:
    input_filename = sys.argv[1]
    try:
        input_device = int(input_filename)
    except:
        pass
else:
    print 'Using default input. Specify a device number to try using your camera, e.g.:'
    print
    print '  python', sys.argv[0], '0'
    print
    input_filename = 'bunny.mp4'

# Choose camera or file, depending upon whether device was set:
if input_device is not None:
    capture = cv2.VideoCapture(input_device)
    if capture:
        print 'Opened camera device number', input_device, '- press Esc to stop capturing.'
else:
    capture = cv2.VideoCapture(input_filename)
    if capture:
        print 'Opened file', input_filename

# Bail if error.
if not capture or not capture.isOpened():
    print 'Error opening video capture!'
    sys.exit(1)

# Fetch the first frame and bail if none.
ok, frame = capture.read()

if not ok or frame is None:
    print 'No frames in video'
    sys.exit(1)

# Now set up a VideoWriter to output video.
w = frame.shape[1]
h = frame.shape[0]

fps = 30

# One of these combinations should hopefully work on your platform:
fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
#fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')

filename = 'captured.'+ext

writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
    print 'Error opening writer'
else:
    print 'Opened', filename, 'for output.'
    writer.write(frame)


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
    
npAllFrames = np.stack(allFrames)
# Find average frame
avgFrame = np.mean(npAllFrames, axis = 0, dtype = 'int32')

cv2.imwrite('avgImage.png', avgFrame)

# calculate differences between each frame and the average
allDiffs = np.absolute(npAllFrames - avgFrame)

# threshold each difference to see what exactly is moving
allThresholds = []
for matrix in allDiffs:
    matrix = matrix.astype('uint8')
    matrix = thresholdFrame(cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY))
    allThresholds.append(matrix)

    # Write if we have a writer.
    if writer:
        writer.write(matrix)

    # Throw it up on the screen.
    cv2.imshow('Video', matrix)
    # Delay for 50ms and get a key
    k = cv2.waitKey(50)



