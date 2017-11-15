import numpy as np

def getAffine(matches):
    v = (0,0)
    for m,n in matches:
        v += m.pt + n.pt
    v = v / len(matches)
    return s
