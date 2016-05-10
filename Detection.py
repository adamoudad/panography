import cv2

# Find Keypoints and Descriptors
def detect(images_list, method="sift"):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_list,descriptors_list = [],[]
    for i in images_list:
        kp,des = sift.detectAndCompute(i,None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    return keypoints_list,descriptors_list
