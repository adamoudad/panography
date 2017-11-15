import cv2

class Matcher:
    def __init__(self,method="BF"):
        self.method = method
        
    def __call__(self,descriptors1,descriptors2):
        return getattr(self,self.method)(descriptors1,descriptors2)
        
    def BF(self,descriptors1,descriptors2):
        """
        Brute Force matching
        Requires 2 descriptors in argument
        """
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(descriptors1,descriptors2)
        # Sort them in the order of their distance.
        return sorted(matches, key = lambda x:x.distance)

    def knnBF(self,descriptors1,descriptors2):
        """
        knn Brute Force
        """
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        return good

    def FLANN(self,descriptors1,descriptors2):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1,descriptors2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        return matches,matchesMask
