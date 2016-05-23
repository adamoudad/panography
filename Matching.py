import cv2

def BruteForce(d1,d2):
    """
    Requires 2 descriptors in argument
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(d1,d2)
    # Sort them in the order of their distance.
    return sorted(matches, key = lambda x:x.distance)

def knnBruteForce(d1,d2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1,d2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    return good

def FLANN(d1,d2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1,d2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    return matches,matchesMask

# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
# # Test if enough matching points have been found
# if len(good)>MIN_MATCH_COUNT:
#     # Calculate transformation between the two images
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()

#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)

#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None
    
# # Draw lines for each pair of matching points
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()
    
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
