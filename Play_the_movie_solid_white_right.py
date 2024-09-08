import cv2
from PIL import ImageGrab
import time
from numpy import ones,vstack
from numpy.linalg import lstsq

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

from PIL import ImageGrab
import time
from numpy import ones,vstack
from numpy.linalg import lstsq

#from . import 02_udacity_photo_line_detection_test


kernel_size = 5
low_threshold = 50
high_threshold = 150



def origin_video():
    cap = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')
    frame_counter = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Our operations on the frame come here
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the resulting frame
        cv2.namedWindow("solidWhiteRight", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("solidWhiteRight", 640,480)
        cv2.imshow("solidWhiteRight",RGB)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

origin_video()
