import numpy as np
import os
import cv2
import numpy as np
import random
import copy

if __name__ == "__main__":

    error_data = '/media/home_bak/ziqi/park/Clean_data/error_data.txt'
    former_data = '/media/home_bak/ziqi/park/Clean_data/former_data.txt'
    correct_file = open(
        "/media/home_bak/ziqi/park/Clean_data/correct_data.txt", "w")
    correct_f = open(error_data)
    data = []
    f1 = open(error_data)

    for line1 in f1:
        line_data1 = line1.strip('\n')
        data.append(line_data1)

    f = open(former_data)
    for line in f:
        line_data = line.strip('\n').split('/')
        if line_data[-1] in data:
            continue
        correct_file.write(line)

    correct_file.close()
