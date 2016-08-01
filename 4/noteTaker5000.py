"""
Owen Kephart & Mayank Agrawal


"""

import cv2
import numpy as np
import sys
import cvk2
from time import sleep

# number of features to use in the feature detector
NUM_MATCHES = 200

def main():
    if len(sys.argv) != 3:
        print "usage: python noteTaker5000.py [static picture with notes] [video perspective]"
        print "e.g. python noteTaker5000.py im1.jpg mov1.mov"
        sys.exit(1)

    # read in command line images
    picName = sys.argv[1]
    vidName = sys.argv[2]

    pic = cv2.imread(picName)
    vid = cv2.VideoCapture(vidName)
    num_frames = int(vid.get(7))
    #num_frames = 120

    h, w, _ = pic.shape

    # area where the notes are written
    note_area_mask = get_note_area_mask(pic)

    # will write the video to an output file
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    outfilename = picName[:picName.find('.')]+'_'+vidName[:vidName.find('.')]+'.avi'
    writer = cv2.VideoWriter(outfilename, fourcc, 30.0, (w, h), True)
    for i in range(num_frames):

        # read in video one frame at a time
        ok, frame = vid.read()
        if not ok or frame is None:
            break

        # print a nice progress bar :)
        print_progress(i+1, num_frames-1)

        # if different shapes, then algorithm won't work
        assert(pic.shape == frame.shape)

        # use feature detection to get get point correspondences, which are
        # then used to create a homography mapping from pic to frame
        H = get_Homography(pic, frame)

        # draw the notes from the static image onto the whiteboard
        noted_whiteboard = note_whiteboard(H, pic, frame, NAM = note_area_mask)

        # write the final output to a video file
        writer.write(noted_whiteboard)
    print ''
    print num_frames
    print 'Your output video is in: ' + outfilename + '.mov'

def note_whiteboard(H, pic, frame, NAM = None):

    h, w, _ = pic.shape

    # if no note_area_mask give, just assume everything is important
    if NAM is None:
        NAM = np.ones((h, w), dtype='uint8')

    # get the text picture warped to look like the board image
    warped_pic = cv2.warpPerspective(pic, H, (w, h))
    warped_pic_gs = cv2.cvtColor(warped_pic.astype('uint8'), cv2.COLOR_BGR2GRAY)

    # use an adaptive threshold to find the notes on the warped whiteboard
    warped_note_mask = cv2.adaptiveThreshold(warped_pic_gs, 255,\
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 3)

    # warped version of the note_area_mask (NAM)
    warped_NAM = cv2.warpPerspective(NAM, H, (w, h))
    warped_NAM_inv = cv2.bitwise_not(warped_NAM)

    # find places where you will color in using the static picture
    #   (so places that are both on the whiteboad, and are darker than their
    #    surroundings are probably notes, so color them in)
    color_mask = cv2.bitwise_and(warped_NAM,warped_note_mask)
    # wherever you should just use the original color
    color_mask_inv = cv2.bitwise_not(color_mask)

    # where you want to use colors from the whiteboard notes
    colors = cv2.bitwise_and(warped_pic, warped_pic, mask=color_mask)
    # where you want to use original video color
    holes = cv2.bitwise_and(frame, frame, mask=color_mask_inv)

    # merge these two together
    return cv2.bitwise_or(colors, holes)

# Has the user specify where on the board the notes are
def get_note_area_mask(pic):

    h, w, _ = pic.shape

    # used to so that image will fit in the screen when selecting points
    scale_factor = int(w/800)

    # instantiate rectangle widget
    areaPicker = cvk2.RectWidget()
    cv2.namedWindow("Note Selector")
    pic_resized = cv2.resize(pic,(w/scale_factor,h/scale_factor))
    areaPicker.start("Note Selector", pic_resized)

    # get corner points
    p1 = (int(areaPicker.points[0][0][0]),int(areaPicker.points[0][0][1]))
    p2 = (int(areaPicker.points[8][0][0]),int(areaPicker.points[8][0][1]))

    # generate the mask for the whiteboard region
    note_area_mask = np.zeros((h/scale_factor,w/scale_factor), dtype='uint8')
    cv2.rectangle(note_area_mask, p1, p2, (255,255,255),-1)

    # get rid of the annoying window
    cv2.destroyWindow("Note Selector")

    # resize the mask
    return cv2.resize(note_area_mask, (w, h))

# Courtesy of Matt Zucker, with some minor modifications
def get_Homography(pic1, pic2):

    # Create feature detector
    detector = cv2.ORB_create()

    # Detect keypoints and compute feature descriptors for both images
    keypoint_arrays = []
    descriptor_arrays = []
    for pic in [pic1, pic2]:
        kp, des = detector.detectAndCompute(pic, None)
        keypoint_arrays.append(kp)
        descriptor_arrays.append(des)

    # Create brute-force matcher to match nearest feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Use the matcher to find the best-matching descriptor in the second
    # array for every descriptor in the first array
    matches = bf.match(descriptor_arrays[0], descriptor_arrays[1])

    # Sort the matches by distance (essentially quality of match)
    matches = sorted(matches, key = lambda x:x.distance)

    # Only take the top n=NUM_MATCHES matches if there are more than that
    n = min(NUM_MATCHES, len(matches))

    point_arrays = []
    for i, kp in enumerate(keypoint_arrays[:n]):
        # Convert KeyPoint object to array of tuples 
        if i == 0:
            pts = [ kp[m.queryIdx].pt for m in matches[:n] ]        
        else:
            pts = [ kp[m.trainIdx].pt for m in matches[:n] ]        

        # Convert to OpenCV-style numpy array
        pts = np.array(pts).reshape((-1, 1, 2))

        # Append to point_arrays
        point_arrays.append(pts)

    # Compute H
    H, _ = cv2.findHomography(point_arrays[0], point_arrays[1], cv2.LMEDS)
    return H

# Adapted from StackOverflow post
def print_progress(iteration, total):
    barLength       = 80
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), 2)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%')),
    sys.stdout.flush()
    if iteration == total:
        print '\nDone!'

main()
