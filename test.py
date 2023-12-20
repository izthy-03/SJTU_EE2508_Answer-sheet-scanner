
import cv2 
import numpy as np
import os
import os.path as osp
from utils import *

path = "./img/cam1.jpg"

img = cv2.imread(path)

b, g, r = cv2.split(img)

red = cv2.merge([0 * b, 0 * g, r])



cv2.imshow("red", red)
cv2.waitKey(0)