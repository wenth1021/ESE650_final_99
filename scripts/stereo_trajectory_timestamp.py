import math
import os
import random
import sys
import cv2
import time

import git
import imageio
import magnum as mn
import numpy as np

file1 = open('report/apartment_3_stereo_CameraTrajectory.txt', 'r')
Lines = file1.readlines()

file2 = open('report/apartment_3_groundtruth.txt', 'r')
groundtruth = file2.readlines()
offset = float(groundtruth[3].split()[0])

file_write = open('report/apartment_3_stereo_CameraTrajectory.txt', 'w')
file_write.truncate(0)

for j in range(len(Lines)):
    spacing = " "
    words = Lines[j].split()
    timestamp = float(words[0]) + offset
    words[0] = format(timestamp, '.6f')
    file_write.write(spacing.join(words) + "\n")

file_write.close()
