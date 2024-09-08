#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

from PIL import ImageGrab
import time
from numpy import ones,vstack
from numpy.linalg import lstsq



kernel_size = 4
low_threshold = 30
high_threshold = 200

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)





def edge_detection():
    while 1:
        # read a original image
        #image = mpimg.imread('test_images/solidWhiteRight.jpg')
        image = np.array(ImageGrab.grab(bbox=(0,40, 800,600)))
        plt.imshow(image)

        # transfer the image into grayscale
        grayscale_img = grayscale(image)
        plt.imshow(grayscale_img,cmap='gray')

        #implement gaussian blur on the image and detect the canny edges
        blur_gray = gaussian_blur(grayscale_img, 5)
        canny_img = canny(blur_gray, 60,160)
        plt.imshow(canny_img,cmap='gray')

        #Your previous comment: add a mask to focus on the region we are interested with (I chose a wrong ROI - Just to show the pipeline flow)
        verticies = np.array([[(0,600),(600,400),(350,365),(450,365)]],dtype=np.int32)
        #verticies = np.array([[(10,650),(10,600),(450,400),(700,400)]], dtype=np.int32)
        img_with_mask = region_of_interest(canny_img,verticies)
        plt.imshow(img_with_mask,cmap='gray')

        # draw hough lines on the masked image
        line_img =  hough_lines(img_with_mask, 2, np.pi/180, 15, 20, 20)
        plt.imshow(line_img)

        #add hough lines into the original image
        line_edges = weighted_img(line_img, image, 0.8, 1, 0)
        #plt.imshow(line_edges)
        #mpimg.imsave('test_images/solidWhiteRight_final.jpg', line_edges)

        cv2.namedWindow('Line_marking', cv2.WINDOW_NORMAL)
        cv2.imshow('Line_marking',line_edges)
        k = cv2.waitKey(10) & 0XFF
        if k == 27:
            break

edge_detection()

