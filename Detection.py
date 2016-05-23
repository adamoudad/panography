import cv2

class Detector:
    def __init__(self,method="SIFT"):
        self.method = method
        
    def __call__(self,image):
        return getattr(self,self.method)(image)
        # return self.__getattr__(self.method)(image)
    
    def SIFT(self,image):
        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(image,None)
