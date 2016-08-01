"""
Owen Kephart & Mayank Agrawal

Will take in an image, and allow the user to select a region centered around
a face. Warps that region into a 256x256 square image.
"""

import cv2
import cvk2
import numpy as np
import sys


def main():
    if len(sys.argv) != 3:
        print "usage: python squareImg.py [input_file] [output_file]"
        print "e.g. python squareImg.py cat.png cat_cropped.png "
        sys.exit(1)

    # read in command line images
    img = cv2.imread(sys.argv[1])

    rw = cvk2.RectWidget(allowRotate=False)

    cv2.namedWindow("Original")
    rw.start("Original",img)

    # select the three upper left corner points to form the triangle for
    # your affine transform
    srcpts = np.array([rw.points[0],rw.points[2],rw.points[6]], np.float32)
    dstpts = np.array([[0, 0], [255, 0], [0, 255]], np.float32)

    at = cv2.getAffineTransform(srcpts,dstpts)

    warped = cv2.warpAffine(img,at,(256,256))

    cv2.imshow("Warped",warped)
    cv2.waitKey(0)

    cv2.imwrite(sys.argv[2],warped)

if __name__ == "__main__":
    main()
