import cv2
from Panorama import Panorama

source_path = "./sources/"
output_path = "./outputs/"
images = [
    cv2.imread(source_path + "1.jpg"),
    cv2.imread(source_path + "2.jpg"),
    # cv2.imread(source_path + "3.jpg"),
    # cv2.imread(source_path + "4.jpg"),
]

p = Panorama(images,matching="knnBF")
p.generate()
p.export(output_path + "panorama.jpg")
