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
