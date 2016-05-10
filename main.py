import cv2

from Detection import detect
from Matching import BruteForce, knnBruteForce, FLANN

source_path = "./sources/"
output_path = "./outputs/"
images = [
    cv2.imread(source_path + "1.jpg"),
    cv2.imread(source_path + "2.jpg"),
]

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# k contains the keypoints for each image, d the descriptors
k,d = detect(images)

img1=cv2.drawKeypoints(images[0],k[0],images[0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2=cv2.drawKeypoints(images[1],k[1],images[1],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output_path + 'sift_keypoints0.jpg',images[0])
cv2.imwrite(output_path + 'sift_keypoints1.jpg',images[1])

# Brute Force Matching
matches = BruteForce(d[0],d[1])
# Draw first 10 matches.
image_matching = cv2.drawMatches(images[0],k[0],images[1],k[1],matches[:10],images[0],flags=2,)
cv2.imwrite(output_path + 'matching_BF.jpg',image_matching)

# Brute Force Matching with knn
matches = knnBruteForce(d[0],d[1])
# cv2.drawMatchesKnn expects list of lists as matches.
image_matching = cv2.drawMatchesKnn(images[0],k[0],images[1],k[1],matches,images[0],flags=2)
cv2.imwrite(output_path + 'matching_knnBF.jpg',image_matching)

# # FLANN matching (Not working)
# matches, matchesMask = FLANN(d[0],d[1])
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# image_matching = cv2.drawMatchesKnn(images[0],k[0],images[1],k[1],matches,None,**draw_params)
# cv2.imwrite(output_path + 'matching_FLANN.jpg',image_matching)
