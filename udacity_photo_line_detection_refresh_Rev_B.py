import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math



from PIL import ImageGrab
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
#from statistics import mean

# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
# Define a kernel size and apply Gaussian smoothing






def edge_detection():
    while 1:
        #last_time = time.time()
        #while True:
        # Read in and grayscale the image
        #image = mpimg.imread('exit-ramp.jpg')
        #image = np.array(ImageGrab.grab(bbox=(900,40,1550,600)))
        image = np.array(ImageGrab.grab(bbox=(0,40, 800,600)))
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)
        ignore_mask_color = 255

        # This time we are defining a four sided polygon to mask
        imshape = image.shape
        vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 1     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 5 #minimum number of pixels making up a line
        max_line_gap = 1    # maximum gap in pixels between connectable line segments
        line_image = np.copy(image)*0 # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
                                                    #B,G,R 순서

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges))

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

        cv2.namedWindow('05_Hough', cv2.WINDOW_NORMAL)
        cv2.imshow('05_Hough',lines_edges)
        k = cv2.waitKey(10) & 0XFF
        if k == 27:
            break



edge_detection()  # 이 파일의 목표는 udacity의 photo인식을 그대로 def함수로 정의하고 그걸 refresh해서 연속영상을 만들어 보는 것이다.
                  #구현은 성공했다. 밑에 waitkey(10)이 핵심, 2번 파일과 짝을 이뤄서 사용해서, 흑백영상에서 edge를 추출해 냈다
                  # 이 파일의 bbox의 영역은 흑백영상이 위치한 스크린 영역이다. 2번파일의 흑백영상에는 지평선 아래 차선만 추출이 되고 있다.
