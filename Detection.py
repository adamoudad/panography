import cv2

# # Find Keypoints and Descriptors
# def sift(images_list):
#     sift = cv2.xfeatures2d.SIFT_create()
#     keypoints_list,descriptors_list = [],[]
#     for i in images_list:
#         kp,des = sift.detectAndCompute(i,None)
#         keypoints_list.append(kp)
#         descriptors_list.append(des)
#     return keypoints_list,descriptors_list

def sift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(image,None)

class Detector:
    def __init__(self,method="SIFT"):
        self.method = method

        def detect(self,image):
            self.__getattr__(self.method)(image)

        def SIFT(self,image):
            sift = cv2.xfeatures2d.SIFT_create()
            return sift.detectAndCompute(image,None)
