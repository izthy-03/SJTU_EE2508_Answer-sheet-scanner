
import cv2 
import numpy as np
import os
import os.path as osp
from utils import *
import pandas
from collections import Counter

path = "./img/cam1.jpg"
csv = "./assets/format.csv"

format = pandas.read_csv(csv, header=None).values

a = "DCAB"
b = "ABCD"
print(Counter(a))
print(Counter(b))
print(Counter(a) == Counter(b))