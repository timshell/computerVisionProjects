"""
Owen Kephart & Mayank Agrawal
1. Blend two images together using Laplacian pyramids
2. Create a hybrid image of two pictures, enabling the
viewer to see image A up close and image B from afar

"""

import cv2
import cvk2
import numpy as np
import sys

sigmaA = 20.0
sigmaB = 1.0
kernel = (17,17)
kA = 0.5
kB = 1.5

def main():
    if len(sys.argv) != 3:
        print "usage: python pyramids.py [pic1] [pic2]"
        print "e.g. python pyramids.py tedCruz.png kevinMalone.png"
        sys.exit(1)

    # read in command line images
    im1 = cv2.imread(sys.argv[1])
    im2 = cv2.imread(sys.argv[2])

    
    # blend the two images
    blended = blended_img(im1,im2)
    cv2.imwrite("blended.png",blended)
    cv2.imshow("", blended.astype("uint8"))
    cv2.waitKey(0)
    

    # create a hybrid image
    hybrid = hybrid_img(im1,im2)
    cv2.imwrite("hybrid.png",hybrid)
    cv2.imshow("", hybrid.astype("uint8"))
    cv2.waitKey(0)

    
def blended_img(im1,im2):

    # have the user draw an ellipse over each image to indicate the regions
    # they wish to have swapped
    ewSrc = cvk2.RectWidget('ellipse')
    ewDst = cvk2.RectWidget('ellipse')

    # get first ellipse
    cv2.namedWindow("FaceA")
    ewSrc.start("FaceA",im1)

    # get second ellipse
    cv2.namedWindow("FaceB")
    ewDst.start("FaceB",im2)

    w, h, _ = im1.shape
    alphaMask = np.zeros((w,h), dtype='float32')

    # get points from these ellipses to generate an affine transform
    srcpts = np.array(ewSrc.points[-4:-1],np.float32)
    dstpts = np.array(ewDst.points[-4:-1],np.float32)

    # create affine transform
    at = cv2.getAffineTransform(dstpts,srcpts)

    cv2.imshow("", cv2.warpAffine(im2,at,(w,h)))
    cv2.waitKey(0)

    # warp the image that will be integrated into the other
    im2 = cv2.warpAffine(im2,at,(w,h))

    # get the center and axes (must make ints because cv2 will get confused)
    center = (int(ewSrc.center[0]),int(ewSrc.center[1]))
    axis = (int(ewSrc.u),int(ewSrc.v))

    # generate the alpha mask
    cv2.ellipse(alphaMask,
                center, 
                axis, 
                ewSrc.angle,
                0,360,(1,1,1),-1)
    # blur the mask to smooth the edges
    alphaMask = cv2.GaussianBlur(alphaMask,(51,51),0)

    # build a pyramid for each image
    lp1 = pyr_build(im1.astype('uint8'))
    lp2 = pyr_build(im2.astype('uint8'))

    # iterate through every pyramid level and blend
    blendedPyramids = []
    for i in range(len(lp1)):
        w, h, _ = lp1[i].shape
        mask = cv2.resize(alphaMask, (w,h), interpolation=cv2.INTER_AREA)
        blendedPyramids.append(alpha_blend(lp1[i], lp2[i], mask))

    return pyr_reconstruct(blendedPyramids)

def pyr_build(pic):
    """
    Generates Laplacian pyramids for a given pic
    
    Input: pic - 8-bit or grayscale image
    Returns: lp - list of pyramids
    """
    depth = 8
    pyrDowns = [pic]
    pyrUps = [None]
    lp = []

    # generate pyrDowns
    for i in range(1, depth+1):
        pyrDowns.append(cv2.pyrDown(pyrDowns[i-1]))

    # generate pyrUps
    for i in range(1, depth + 1):
        w, h, _ = pyrDowns[i-1].shape
        pyrUps.append(cv2.pyrUp(pyrDowns[i], None, (w, h)))
    
    for i in range(depth):
        temp = pyrDowns[i].astype('float32') - pyrUps[i+1].astype('float32')
        lp.append(temp)

    lp.append(pyrDowns[depth].astype("float32"))

    return lp

def pyr_reconstruct(lp):
    """
    Reconstructs a picture given its laplacian pyramids
    
    Input: lp - list of laplacian pyramids
    Returns: None (just displays picture)
    """
    temps = []
    n = len(lp)

    # initialize list
    for item in lp:
        temps.append(None)
    temps[n-1] = lp[n-1]

    # work backwards to reconstruct
    for i in range(n-2, -1, -1):

        w, h, _ = lp[i].shape
        temps[i] = cv2.pyrUp(temps[i+1], None, (w, h)) + lp[i]

    temps[0] = (np.clip(temps[0], 0, 255)) # handles overflow
    return temps[0].astype('uint8')


def alpha_blend(A, B, alpha):

    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)
    C = (A + alpha*(B-A))
    return C

def hybrid_img(A,B):   

    A = cv2.cvtColor(A,cv2.COLOR_RGB2GRAY)
    B = cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)

    Apass = cv2.GaussianBlur(A.astype('float32'),kernel,sigmaA)
    Bpass = B.astype('float32') - \
        cv2.GaussianBlur(B.astype('float32'),kernel,sigmaB)
    
    return np.clip(kA*Apass + kB*Bpass, 0, 255)

if __name__ == "__main__":
    main()


