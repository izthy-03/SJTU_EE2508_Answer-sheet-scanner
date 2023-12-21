
import cv2 
import numpy as np
import os
import os.path as osp
from utils import *
import pandas

path = "./img/cam1.jpg"
csv = "./assets/format.csv"

format = pandas.read_csv(csv, header=None).values

print(format[16])