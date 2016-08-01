"""
Owen Kephart & Mayank Agrawal


"""

import cv2
import numpy as np
import sys
import cvk2


def main():
    if len(sys.argv) != 3:
        print "usage: python noteTaker3000.py [skewed perspective with text] [straight perspective]"
        print "e.g. python noteTaker3000.py im1.jpg im2.jpg"
        sys.exit(1)

    # read in command line images
    skewedText = cv2.imread(sys.argv[1])
    straightClean = cv2.imread(sys.argv[2])

    assert(skewedText.shape == straightClean.shape)

    # Create feature detector
    detector = cv2.ORB_create()

    # Detect keypoints and compute feature descriptors for both images
    keypoint_arrays = []
    descriptor_arrays = []

    for pic in [skewedText, straightClean]:

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

    #print 'got', len(matches), 'matches'

    # Only take the top n=10 matches if there are more than that
    n = min(100, len(matches))

    # Draw matches
    """
    match_display = cv2.drawMatches(skewedText, keypoint_arrays[0],
                                    straightClean, keypoint_arrays[1],
                                    matches[:n], outImg=None, flags=2)

    label_image(match_display, 'Keypoints matched by descriptor - '
                'hit any key to start interactive demo...')

    win = 'Fundamental matrix explorer'

    cv2.namedWindow(win)
    cv2.moveWindow(win, 0, 0)
    cv2.imshow(win, match_display)
    while cv2.waitKey(5) < 0: pass
    """

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

    warped = cv2.warpPerspective(skewedText, H, (1080,720))
    warpedGray = cv2.cvtColor(warped.astype('uint8'), cv2.COLOR_BGR2GRAY)
    light_mask = cv2.adaptiveThreshold(warpedGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 3, 3)
    light_mask = cv2.erode(light_mask,np.ones((2,2),np.uint8),iterations = 1)
    
    dark_mask = cv2.bitwise_not(light_mask)

    areaPicker = cvk2.RectWidget()
    cv2.namedWindow("Note Selector")
    areaPicker.start("Note Selector",straightClean)

    w, h, _ = straightClean.shape
    draw_mask = np.zeros((w,h), dtype='uint8')
    # get the center and axes (must make ints because cv2 will get confused)
    center = (int(areaPicker.center[0]),int(areaPicker.center[1]))
    axis = (int(areaPicker.u),int(areaPicker.v))

    print areaPicker.points
    print areaPicker.points[0]
    p1 = (int(areaPicker.points[0][0][0]),int(areaPicker.points[0][0][1]))
    p2 = (int(areaPicker.points[8][0][0]),int(areaPicker.points[8][0][1]))
    print p1
    print p2

    # generate the mask for the whiteboard region
    cv2.rectangle(draw_mask, p1, p2, (255,255,255),-1)
    draw_mask_inv = cv2.bitwise_not(draw_mask)

    color_mask = cv2.bitwise_and(draw_mask,dark_mask)
    hole_mask = cv2.bitwise_and(draw_mask,light_mask)

    colors = cv2.bitwise_and(warped, warped, mask=color_mask)
    holes = cv2.bitwise_and(straightClean, straightClean, mask=hole_mask)

    colored = cv2.bitwise_or(colors, holes)
    background = cv2.bitwise_and(straightClean, straightClean, mask=draw_mask_inv)

    final = cv2.bitwise_or(background, colored)

    cv2.imshow("test", final)
    cv2.waitKey(0)

def label_image(image, text):
    
    h = image.shape[0]
    
    cv2.putText(image, text, (8, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,0), 3, cv2.LINE_AA)

    cv2.putText(image, text, (8, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 1, cv2.LINE_AA)


main()
