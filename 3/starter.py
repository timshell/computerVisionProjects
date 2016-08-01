#!/usr/bin/env python

import cv2
import numpy
import sys
import os
from PointCloudApp import * 

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
window_size = 21
param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity, 
                                max_disparity, 
                                window_size, 
                                param_P1, 
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparitieactualPoints.
disparity = matcher.compute(cam_image, proj_image) / 16.0

# Pop up the disparity image.
#cv2.imshow('Disparity', disparity/disparity.max())
#while cv2.waitKey(5) < 0: pass

# matrix for intrinsic params
f = 600
u0 = 320
v0 = 240
K = numpy.vstack([[f, 0, u0], [0, f, v0], [0, 0, 1]])
Kinv = numpy.linalg.inv(K)

stereoBaseline = 0.05
maxZ = 8

rows = proj_image.shape[0]
cols = proj_image.shape[1]

# create mesh grid for coordinates
u = numpy.linspace(0, cols-1, cols)
v = numpy.linspace(0, rows-1, rows)

uVals, vVals = numpy.meshgrid(u, v)

# pick out non-zero cam pixels
uVals = uVals
vVals = vVals
zVals = numpy.ones((rows,cols))


# create a list of form [u, v, 1] of all non-zero cam pixels
allQ = numpy.stack([uVals,vVals,zVals], axis = 1)

# map these points through Kinv
points = numpy.transpose(numpy.dot(Kinv,numpy.transpose(allQ)))

# get all disparities above threshold
disparity_mask = disparity > f*stereoBaseline/maxZ
disparity = disparity[disparity_mask]

# set up empty arrays for these values
ratio   = numpy.zeros_like(disparity)
actualX = numpy.zeros_like(disparity)
actualY = numpy.zeros_like(disparity)
actualZ = numpy.zeros_like(disparity)


# find actual Z values
actualZ = f*stereoBaseline/disparity

# get all points that don't have a 0 Z value 
points = points[disparity_mask]
point_mask = points[:,2] != 0

# calculate ratio to find scaling
ratio = actualZ[point_mask]/(points[:,2][point_mask])

# use ratio to scale properly
actualX = ratio*points[:,0]
actualY = ratio*points[:,1]

# generate list of actual points
actualPoints = numpy.stack([actualX,actualY,actualZ],axis=1)

# generate image
app = PointCloudApp(actualPoints, allow_opengl=True)
app.run()


















