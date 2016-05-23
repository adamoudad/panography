import cv2
import numpy as np
import Matching
import Detection

class Panorama:
    def __init__(self,images):
        self.min_matches = 10
        self.source = images
        self.output = None
        self.descriptors = None

    def generate(self):
        self.output = self.source[0]
        for image in self.source[1:]:
            self.keypoints, self.descriptors = self.detect(self.output)
            k_source,d_source = self.detect(self.source[1])
            matches = self.match(self.descriptors,d_source)
            self.stitch(image,k_source,matches)
            self.crop()

    def stitch(self,image,keypoints,matches):
        if len(matches)>self.min_matches:
            pts_0 = np.float32([ self.keypoints[m.queryIdx].pt for m in matches ])
            pts_1 = np.float32([ keypoints[m.trainIdx].pt for m in matches ])
            
            M, mask = cv2.findHomography(pts_1, pts_0, cv2.RANSAC,5.0)
            
            result = cv2.warpPerspective(image,M,(self.output.shape[1]+image.shape[1],max(self.output.shape[0],image.shape[0])))
            result[:self.output.shape[0],:self.output.shape[1]] = self.output 
            self.output = result
        else:
            print("Not enough matches are found - {0}/{1}".format(len(matches),self.min_matches))

    def detect(self,image, method="SIFT"):
        return Detection.sift(image)
        
    def match(self,descriptors1,descriptors2,method="BF"):
        return Matching.BruteForce(descriptors1,descriptors2)
        # matches = knnBruteForce(d[0],d[1])
        # # FLANN matching (Not working)
        # matches, matchesMask = FLANN(d[0],d[1])
        # draw_params = dict(matchColor = (0,255,0),
        #                    singlePointColor = (255,0,0),
        #                    matchesMask = matchesMask,
        #                    flags = 0)
        # image_matching = cv2.drawMatchesKnn(images[0],k[0],images[1],k[1],matches,None,**draw_params)
        # cv2.imwrite(output_path + 'matching_FLANN.jpg',image_matching)
        
    def export(self,path):
        if self.output is not None:
            cv2.imwrite(path,self.output)
        else:
            print("Please generate the panorama first.")

    def crop(self):
        """
        Crop black edges of the resulting panorama
        """
        if self.output is not None:
            gray = cv2.cvtColor(self.output,cv2.COLOR_BGR2GRAY)
            _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
            _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            self.output = self.output[y:y+h,x:x+w]
        else:
            print("Please generate the panorama first.")
