"""
Detects Lane lines using HLS and Grays Scale thresholding, Sobel Filter, Canny Edge Detection, and Eroding Filter
"""

import cv2 as cv 
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema


class Video():

    def __init__(self, cap, fps=30, isColor=True, name="V0.2"):
        self.cap = cv.VideoCapture(cap)
        self.name = name
        self.width =  int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame = np.zeros((self.width,self.height, 3), dtype=np.uint8) 
        self.areaInterestPoints =  [(self.width, 94*self.height/100),
                                    (0, 94*self.height/100),
                                    (0, 73*self.height/96),
                                    (2*self.width/5, 10*self.height/20),
                                    (3*self.width/5, 10*self.height/20),
                                    (self.width, 73*self.height/96),
                                    (self.width, 94*self.height/100)]

        self.perspectivePoints = np.array([[2*self.width/5, 10*self.height/20],
                                           [3*self.width/5, 10*self.height/20],
                                           [self.width, 7*self.height/9],
                                           [0, 7*self.height/9]], dtype = "float32")
        self.perspectivePointsOrdered = None
        self.fps = fps
        self.isColor = isColor
        self.temp_img = None
        self.og_frame = None
        self.gray = None
        self.i = 0
        self.maxm = np.array([])
        self.maxLeft = np.array([])
        self.maxRight = np.array([])
        self.objects = []
        self.buffer = []
        self.buffer_length = 4 #for moving average of straight lines
        self.max_lost_frames = 30
        self.consec_lost = 0


    def display(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, self.frame = self.cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            self.og_frame = self.frame
            #process frame
            self.process_linear()
            # Display the resulting frame
            cv.imshow(self.name, self.frame)
            cv.imshow("Original", self.og_frame)
            
            #delays the video being played
            cv.waitKey(int((1000/self.fps)+0.5)) #for __ FPS  -> 1000/FPS = X milliseconds 

            if cv.waitKey(1) == 27 or cv.waitKey(1) == ord('q'):
                # ESC pressed
                break
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()

    def process_linear(self):
        """
        Runs the pipline on the fed image
        """
        self.resize()
        self.mask_lanes_alt()
        self.to_binary()
        self.road_mask()
        self.hough_lines()
    
    def resize(self, width=480, height=360):
        """
        takes an image resize it
        assigns height and width for future operations and recalculates area of interest and perspective points
        """
        self.frame = cv.resize(self.frame,(width,height))
        self.og_frame = self.frame
        self.width =  width
        self.height = height
        self.areaInterestPoints =  [(self.width, 94*self.height/100),
                                    (0, 94*self.height/100),
                                    (0, 73*self.height/96),
                                    (2*self.width/5, 10*self.height/20),
                                    (3*self.width/5, 10*self.height/20),
                                    (self.width, 73*self.height/96),
                                    (self.width, 94*self.height/100)]

        self.perspectivePoints = np.array([[2*self.width/5, 10*self.height/20],
                                           [3*self.width/5, 10*self.height/20],
                                           [self.width, 7*self.height/9],
                                           [0, 7*self.height/9]], dtype = "float32")

    def to_gray(self):
        """
        turns frame from color to gray scale
        """
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

    def hough_lines(self):
        """
        detects hough lines
        splits lines to left and right (from left half screen and right half screen)
        creates average frame line (yellow) and extrapolates
        creates n-frame moving average (blue) line to have a smother line than the yellow
        """
        line_left_update = False
        line_right_update = False

        endY = 17*self.height/18
        startY = self.height/2

        #check buffer length and removes index 0 item if too long
        if len(self.buffer) > self.buffer_length:
            self.buffer.pop(0)


        #lines stores line vectors min line legthe was 50
        lines = cv.HoughLinesP(self.frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=20,maxLineGap=30)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(self.og_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)
 
        #creates the yellow extrapolated lines
        if lines is not None:
            self. consec_lost = 0 #consecutively lost frames
            line_right = np.array([[0,0,0,0]])
            line_left = np.array([[0,0,0,0]])
            img_center = self.frame.shape[1]//2
            #separates lines into left and right 
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1<img_center and x2<img_center:
                        line_left = np.concatenate((line_left,np.array([[x1, y1, x2,y2]])), axis=0)
                    elif x1>img_center and x2>img_center:
                        line_right = np.concatenate((line_right, np.array([[x1, y1, x2,y2]])), axis=0)
                    else:
                        continue
            line_right = np.delete(line_right,0,0)
            line_left = np.delete(line_left,0,0)

            #creates right side extrapolated line by averaging all the line positions and slopes
            #average x1,x2,y1,y2 for right and left
            if not np.isnan(line_right).any() and line_right.size > 0:
                #average of lines
                line_right = np.mean(line_right,axis=0)
                self.og_frame = cv.line(self.og_frame, (int(line_right[0]), int(line_right[1])), (int(line_right[2]), int(line_right[3])), (0,255,0), 3, cv.LINE_AA)
                #now extrapolate line // make the line longer
                slope = (line_right[1]-line_right[3])/(line_right[0]-line_right[2])
                if slope != 0:
                    intercept = line_right[1] - (slope * line_right[0])
                    startX = (startY - intercept) / slope
                    endX = (endY - intercept) / slope
                    line_right = np.array([endX,endY,startX,startY])
                    if not np.isnan(line_left).any():
                        self.og_frame = cv.line(self.og_frame, (int(line_right[0]), int(line_right[1])), (int(line_right[2]), int(line_right[3])), (0,255,255), 3, cv.LINE_AA)
                        line_right_update = True
            if not line_right_update and len(self.buffer)>0:
                #if no right lines seen use last buffer frame
                i = len(self.buffer) - 1
                self.og_frame = cv.line(self.og_frame, (int(self.buffer[i][0]), int(self.buffer[i][1])), (int(self.buffer[i][2]), int(self.buffer[i][3])), (0,255,255), 3, cv.LINE_AA)
                line_right = np.array([self.buffer[i][4], self.buffer[i][5], self.buffer[i][6], self.buffer[i][7]])
                print("Warning: Dropped Right Lines")

            #creates left side extrapolated line by averaging all the line positions and slopes
            if not np.isnan(line_left).any() and line_left.size > 0:
                #average Line
                line_left = np.mean(line_left,axis=0)

                slope = (line_left[1]-line_left[3])/(line_left[0]-line_left[2])
                if slope != 0:
                    intercept = line_left[1] - (slope * line_left[0])
                    startX = (startY - intercept) / slope
                    endX = (endY - intercept) / slope
                    line_left = np.array([endX,endY,startX,startY])
                    if not np.isnan(line_left).any():
                        self.og_frame = cv.line(self.og_frame, (int(line_left[0]), int(line_left[1])), (int(line_left[2]), int(line_left[3])), (0,255,255), 3, cv.LINE_AA)
                        line_left_update = True
            if not line_left_update and len(self.buffer)>0:
                #if no lines seen use last buffer frame
                i = len(self.buffer) - 1
                self.og_frame = cv.line(self.og_frame, (int(self.buffer[i][4]), int(self.buffer[i][5])), (int(self.buffer[i][6]), int(self.buffer[i][7])), (0,255,255), 3, cv.LINE_AA)
                line_left = np.array([self.buffer[i][0], self.buffer[i][1], self.buffer[i][2], self.buffer[i][3]])
                print("Warning: Dropped Left Lines")

            #add left x2,y2,x1,y1 , right x2,y2,x1,y1
            #updates buffer if there are detected line
            if line_right.size > 0 and line_left.size > 0:
                self.buffer.append([int(line_left[0]), int(line_left[1]), int(line_left[2]), int(line_left[3]),int(line_right[0]), int(line_right[1]), int(line_right[2]), int(line_right[3])])
            else:
                print("Warning: buffer not updated")
        
        #draw moving average lines (blue line)
        if len(self.buffer) >= self.buffer_length and self.consec_lost < self.max_lost_frames:
            self.consec_lost += 1
            x = np.mean(np.asarray(self.buffer),axis=0)
            self.og_frame = cv.line(self.og_frame, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (255,0,0), 3, cv.LINE_AA)
            self.og_frame = cv.line(self.og_frame, (int(x[4]), int(x[5])), (int(x[6]), int(x[7])), (255,0,0), 3, cv.LINE_AA)
        else:
            if self.consec_lost > self.max_lost_frames:
                print("Error: Too many lost frames")
            else:
                print("Warning: Buffer too small MA not drawn")

    def road_mask(self):
        """
        masks image so that only he important pixels are used in detecting the lane lines 
        """
        mask = np.zeros_like(self.frame) #makes blank image
        match_mask_color = 255
        cv.fillPoly(mask, np.array([self.areaInterestPoints], dtype=np.int32), match_mask_color)
        masked_image = cv.bitwise_and(self.frame, mask)
        self.frame = masked_image


    def to_binary(self):
        """
        turns frame to binary
        """
        ret3,self.frame = cv.threshold(self.frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    def blur(self):
        """
        blurs image using gaussian blur
        """
        self.frame = cv.GaussianBlur(self.frame, ksize=(5,5), sigmaX=0,sigmaY=0)
    
    def mask_lanes_alt(self):
        """
        masks the lane lines:
        1. using HLS and gray scale thresholding
        2. Using Sobel filter
        3. Canny Edge detection
        4. using an eroding function
        """
        #color threshold

        converted = cv.cvtColor(self.frame, cv.COLOR_BGR2HLS)
        #white color mask
        lower = np.uint8([0,200,0])
        upper = np.uint8([200,255,255])
        white_mask = cv.inRange(converted, lower, upper)
        #yellow mask
        lower = np.uint8([10,0,100])
        upper = np.uint8([40,255,255])
        yellow_mask = cv.inRange(converted, lower, upper)
        #combine the mask
        mask = cv.bitwise_or(white_mask, yellow_mask)
        self.temp_img = cv.bitwise_and(self.frame, self.frame, mask = mask)
        
        self.temp_img = cv.cvtColor(self.temp_img, cv.COLOR_BGR2GRAY)

         #temp frame for canny
        frameTemp = self.frame

        x = 3
        y = 1
        x_out = cv.CV_16S
        y_out = cv.CV_16S
        scale = 1
        delta = 0
        kernal = np.ones((5,5),np.uint8)
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray, x_out, 1, 0, ksize=x, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) #size was 3
        grad_y = cv.Sobel(gray, y_out, 0, 1, ksize=y, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) #size was 3

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        
        #combine sobel x and y
        self.frame = cv.addWeighted(abs_grad_x, 2, abs_grad_y, 1, 0)

        #canny edge dectetion
        frameTemp = cv.Canny(frameTemp,140,200)
        
        #adding canny and sobel together
        self.frame = cv.add(self.frame,frameTemp)

        kernel = np.ones((2,2),np.uint8)

        #eroding image  NOTE: Erosion does a lot more hurting than helping
        self.frame = cv.erode(self.frame,kernel, iterations=1)

        #adding canny/sobel with color threshold
        self.frame = cv.add(self.frame, self.temp_img)

if __name__ == "__main__":
    """
    to use just create a Video object and pass a string to the file of the video you which to use
    you will likely need to tweak the perspective points as the masking process is important from video to video (unless filming from same spot)
    """
    test2 = Video('test_videos/test1.mp4')
    test2.display()

