"""
Owen Kephart & Mayank Agrawal


"""

import cv2
import numpy as np
import sys
import cvk2


def main():
    if len(sys.argv) != 4:
        print "usage: python noteTaker3000.py [skewed perspective] [skewed perspective with text] [straight perspective]"
        print "e.g. python noteTaker3000.py im1.jpg im2.jpg im3.jpg "
        sys.exit(1)

    # read in command line images
    skewedClean = cv2.imread(sys.argv[1])
    skewedText = cv2.imread(sys.argv[2])
    straightClean = cv2.imread(sys.argv[3])

    assert((skewedClean.shape == skewedText.shape) and (skewedText.shape == straightClean.shape))

    # Create feature detector
    detector = cv2.ORB_create()

    # Detect keypoints and compute feature descriptors for both images
    keypoint_arrays = []
    descriptor_arrays = []

    for pic in [skewedClean, straightClean]:

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
    n = min(25, len(matches))

    # Draw matches
    match_display = cv2.drawMatches(skewedClean, keypoint_arrays[0],
                                    straightClean, keypoint_arrays[1],
                                    matches[:n], outImg=None, flags=2)

    label_image(match_display, 'Keypoints matched by descriptor - '
                'hit any key to start interactive demo...')

    win = 'Fundamental matrix explorer'

    cv2.namedWindow(win)
    cv2.moveWindow(win, 0, 0)
    cv2.imshow(win, match_display)
    while cv2.waitKey(5) < 0: pass

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

    w,h = skewedClean.shape[:2]
    p = np.array( [ [[0, 0]],
                    [[w, 0]],
                    [[w, h]],
                    [[0, h]] ], dtype='float32' )

    # Map through warp
    pp = cv2.perspectiveTransform(p, H)

    # Get integer bounding box of form (x0, y0, width, height)
    box = cv2.boundingRect(pp)
    # Separate into dimensions and origin
    origin = box[0:2]
    dims = box[2:4]

    # Create translation transformation to shift images
    T = np.eye(3)
    T[0,2] -= origin[0]
    T[1,2] -= origin[1]

    # Compose homography and translation via matrix multiplication

    Hnice = np.matrix(T) * np.matrix(H)


    warp = cv2.warpPerspective(skewedClean, H, straightClean.shape[:2])
    #cv2.imshow("test", warp)
    #cv2.waitKey(0)


    grayText = cv2.cvtColor(skewedText.astype('uint8'), cv2.COLOR_BGR2GRAY)
    grayClean = cv2.cvtColor(skewedClean.astype('uint8'), cv2.COLOR_BGR2GRAY)

    diff = cv2.subtract(grayClean, grayText)
    warpedDiff = cv2.warpPerspective(diff, H, (1080,720))

    ret, mask = cv2.threshold(warpedDiff, 20, 255, cv2.THRESH_BINARY)
    ret, invMask = cv2.threshold(warpedDiff, 20, 255, cv2.THRESH_BINARY_INV)

    warpedText = cv2.warpPerspective(skewedText, H, (1080,720))

    colors = cv2.bitwise_and(warpedText, warpedText, mask = mask)
    cv2.imshow("test", colors)
    cv2.waitKey(0)

    holes = cv2.bitwise_and(straightClean, straightClean, mask = invMask)

    final = cv2.bitwise_or(colors, holes)
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