# Panography
Small Computer Vision project in Python 3 with OpenCV 3 for image stitching in panography

This code can create a panorama from a list of images.

* [How it works](#howitworks)  
* [Usage](#usage)
* [Note](#note)

<a name="howitworks"/>
# How it works
The class Panorama manage the list of images to be stitched into one full panorama.
The three main steps are :
* Finding Keypoints and Descriptors (Detection)
* Matching Keypoints (Matching)
* Stitching

This is done with the first two images, then from this intermediate panorama, the next images are stitched following the same process.

## Detection
The SIFT detection is used to find keypoints and describe the images.
Next improvements will provide other algorithms

## Matching
Brute Force and Brute Force Knn are the two (most basic) algorithms used for matching keypoints.

## Stitching
It encompasses as well finding the homography transformation, and computing this transformation. Stitching is then done by appending the reference image for the homography.

<a name="usage"/>
# Usage
You need to create two directories for the source and output images :
* ./sources/ for the images that will be stitched into a panorama
* ./outputs/ for the output panorama

Then just change the main.py to your convenience.

<a name="note"/>
# Note
I try to provide a code easy to read and concise. Also, please feel free to tell me some weird things you might see in there ;).
