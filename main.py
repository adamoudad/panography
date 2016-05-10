import cv2
import numpy as np

from Matcher import BruteForce

source_path = "./sources/"
output_path = "./outputs/"
images = [
    cv2.imread(images_path + "1.jpg"),
    cv2.imread(images_path + "2.jpg"),
]

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Find Keypoints and Descriptors
def keypoints_descriptors(images_list):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_list,descriptors_list = [],[]
    for i in images_list:
        kp,des = sift.detectAndCompute(i,None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    return keypoints_list,descriptors_list

k,d = keypoints_descriptors(images)

img1=cv2.drawKeypoints(images[0],k[0],images[0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2=cv2.drawKeypoints(images[1],k[1],images[1],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output_path + 'sift_keypoints0.jpg',images[0])
cv2.imwrite(output_path + 'sift_keypoints1.jpg',images[1])



# Brute Force Matching
matches = BruteForce(d[0],d[1])

# Draw first 10 matches.
img3 = cv2.drawMatches(images[0],k[0],images[1],k[1],matches[:10],images[0],flags=2,)
cv2.imwrite(output_path + 'matching.jpg',img3)

print(matches)


# # FINDING BEST MATCHES
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
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
    

# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)





