import cv2
import numpy as np
import Matching
import Detection

class Panorama:
    def __init__(self):
        self.min_matches = 10
        self.panorama = None
    def stitch(self,images):
        k,d = self.detect(images)
        matches = self.match(d)
        if len(matches)>self.min_matches:
            pts_0 = np.float32([ k[0][m.queryIdx].pt for m in matches ])
            pts_1 = np.float32([ k[1][m.trainIdx].pt for m in matches ])
        
            M, mask = cv2.findHomography(pts_1, pts_0, cv2.RANSAC,5.0)
        
            result = cv2.warpPerspective(images[1],M,(images[0].shape[1]+images[1].shape[1],max(images[0].shape[0],images[1].shape[0])))
            result[:images[0].shape[0],:images[0].shape[1]] = images[0] 

            self.output = result
            
        else:
            print("Not enough matches are found - {0}/{1}".format(len(matches),MIN_MATCH_COUNT))

    def detect(self,images, method="SIFT"):
        return Detection.sift(images)
        
    def match(self,descriptors,method="BF"):
        return Matching.BruteForce(descriptors[0],descriptors[1])
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
            print("Panorama is not generated")
