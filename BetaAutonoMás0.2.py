"""
5-25-20 - working on setting up sliders to tweak hough lines, sobel filter, erode and dilate, 
"""

import cv2 as cv 
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema


class Video():

    def __init__(self, cap, fps=30, isColor=True, name="V0.2", cascade='HarrCascades/cars.xml'):
        self.cap = cv.VideoCapture(cap)
        self.name = name
        self.width =  int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame = np.zeros((self.width,self.height, 3), dtype=np.uint8) #initilize frame is banl(0's)
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
        self.cascade = cv.CascadeClassifier(cascade)
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
            #self.process()
            self.process_linear()
            # Display the resulting frame
            cv.imshow(self.name, self.frame)
            cv.imshow("Original", self.og_frame)
            #delays the video being played

            cv.waitKey(int((1000/self.fps)+0.5)) #for __ FPS  -> 1000/FPS = X miliseconds 

            if cv.waitKey(1) == 27 or cv.waitKey(1) == ord('q'):
                # ESC pressed
                break
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()

    def process_linear(self):
        self.resize()
        #self.mask_lanelines()
        self.mask_lanes_alt()
        self.to_binary()
        self.road_mask()
        #self.object_rec()
        self.hough_lines()

    def process(self):
        self.transform()
        self.mask_lanelines()
        self.blur()
        self.to_binary()
        self.count_pixels()
        #self.sliding_windows()
        #self.to_gray()
        #self.road_mask()
    
    def resize(self, width=480, height=360):
        """
        takes an image resize it
        asigns height and width for future operations and recalcs area of interes and perspective points
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
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

    def to_color(self):
        self.frame = cv.cvtColor(self.frame, cv.COLOR_GRAY2BGR)

    def hough_lines(self):
        """
        detects hough lines
        splits lines to left and right (from left half screen and right half screen)
        creates average frame line (yellow) and extrapoates
        creates n-frame moving average (blue) line
        """
        line_left_update = False
        line_right_update = False

        endY = 17*self.height/18
        startY = self.height/2

        #check buffer length
        if len(self.buffer) > self.buffer_length:
            self.buffer.pop(0)


        #lines stores line vectors min line legthe was 50
        lines = cv.HoughLinesP(self.frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=20,maxLineGap=30)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(self.og_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)
 
        #TODO: Average out hough lines and graph
        #so somehow seperate lines to left and right
        #   -easy: devide screen in half is throwing bugs around curves 
        #   -harder: idk think
        #(x0,y0,x1,y1)
        if lines is not None:
            self. consec_lost = 0 #consecutavely lost frames
            line_right = np.array([[0,0,0,0]])
            line_left = np.array([[0,0,0,0]])
            img_center = self.frame.shape[1]//2
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1<img_center and x2<img_center:
                        line_left = np.concatenate((line_left,np.array([[x1, y1, x2,y2]])), axis=0)
         #               print(y1)
                    elif x1>img_center and x2>img_center:
                        line_right = np.concatenate((line_right, np.array([[x1, y1, x2,y2]])), axis=0)
                    else:
                        continue
            line_right = np.delete(line_right,0,0)
            line_left = np.delete(line_left,0,0)

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
                    line_right = np.array([endX,endY,startX,startY]) # really has to be [startX,startY,endX,endY]
                    if not np.isnan(line_left).any():
                        self.og_frame = cv.line(self.og_frame, (int(line_right[0]), int(line_right[1])), (int(line_right[2]), int(line_right[3])), (0,255,255), 3, cv.LINE_AA)
                        line_right_update = True
            if not line_right_update and len(self.buffer)>0:
                #if no right lines seen use last buffer frame
                i = len(self.buffer) - 1
                self.og_frame = cv.line(self.og_frame, (int(self.buffer[i][0]), int(self.buffer[i][1])), (int(self.buffer[i][2]), int(self.buffer[i][3])), (0,255,255), 3, cv.LINE_AA)
                line_right = np.array([self.buffer[i][4], self.buffer[i][5], self.buffer[i][6], self.buffer[i][7]])
                print("Warning: Dropped Right Lines")
            #else:
            #    #not line to begin with
            #    print("Warning: no intial line")

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
        
        #draw moving average lines
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
        mask = np.zeros_like(self.frame) #makes blank image
        match_mask_color = 255
        cv.fillPoly(mask, np.array([self.areaInterestPoints], dtype=np.int32), match_mask_color)
        masked_image = cv.bitwise_and(self.frame, mask)
        self.frame = masked_image

    def count_pixels(self):
        """
        takes in single channel image (gray, binary)
        """
        #print(self.frame.shape)   #frame[row(y),col(x)]
        
        #create 1D array with number of __ pixels in the column
        pixels = np.array([])
        for i in range(0,self.frame.shape[1]):
            pixels = np.append(pixels,cv.countNonZero(self.frame[:,i]))

        #maxm holds x values of graph maximums
        x = np.arange(pixels.shape[0])
        y = np.polyfit(x, pixels, 4)
        p = np.poly1d(y)
        self.maxm = argrelextrema(p(x), np.greater)
        
        #plot
        #if  self.i % 1 == 0:
            #x = np.arange(pixels.shape[0])
            #y = np.polyfit(x, pixels, 4)
            #p = np.poly1d(y)
            #graph = plt.plot(x,p(x))
            #maxm holds x values of graph maximums
            #self.maxm = argrelextrema(p(x), np.greater)
            #plt.show()
        #self.i += 1
        print(self.i)

    def sliding_windows(self):
        """
        sliding window search
        """
        num_boxes = 10
        box_width = 10
        #window size
        w_window, h_window = (int(self.frame.shape[1]/box_width), int(self.frame.shape[0]/num_boxes))
        #array to hold list of aproximate coordinates of lane
        #left_lane_coords_x = np.array([])
        #right_lane_coords_x = np.array([])

        #lane_coords_y = np.array([])
        # Empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Identify the x-y positions of all nonzero pixels in the image
        # Note: the indices here are equivalent to the coordinate locations of the
        # pixel
        nonzero = self.frame.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        print(self.maxm)
        #try:
        left_x1,left_y1 = int(self.maxm[0][0]-(w_window/2.0)), self.frame.shape[0]-h_window
        left_x2, left_y2 = int(self.maxm[0][0]+(w_window/2.0)), self.frame.shape[0]

        right_x1,right_y1 = int(self.maxm[0][1]-(w_window/2.0)), self.frame.shape[0]-h_window
        right_x2, right_y2 = int(self.maxm[0][1]+(w_window/2.0)), self.frame.shape[0]

        for box in range(0,num_boxes):
            #draw box and do operations for next box loacation and save points
            #1. Draw/Make Box    TODO: test windows are in the right orinetation
            left_window = self.frame[left_y1:left_y2, left_x1:left_x2]
            right_window = self.frame[right_y1:left_y2, right_x1:right_x2]

            left_lane_inds.append(left_window.nonzero()[0])
            right_lane_inds.append(right_window.nonzero()[0])

            print(nonzero)

            #2. Calculate mean (center of pixel mass) of x-axis
            #left_px = np.array([])
            #right_px = np.array([])
            #for i in range(0,left_window.shape[1]):
            #    left_px = np.append(left_px,cv.countNonZero(left_window[:,i]))
            #    right_px = np.append(right_px,cv.countNonZero(right_window[:,i]))

            #xR = np.arange(right_px.shape[0])
            #yR = np.polyfit(xR, right_px, 3)
            #pR = np.poly1d(yR)

            #xL = np.arange(left_px.shape[0])
            #yL = np.polyfit(xL, left_px, 3)
            #pL = np.poly1d(yL)
            #maxm holds x values of graph maximums
            #self.maxRight = argrelextrema(pR(xR), np.greater)
            #self.maxLeft = argrelextrema(pL(xL), np.greater)

            #graphR = plt.plot(xR,pR(xR))
            #graphL = plt.plot(xL,pL(xL))
            #plt.show()
            """
            #3. Update for next boxes X and Y if there were pixels in image TODO: add threshold for moving right now threhod is 0
            if self.maxLeft[0].size != 0 and (left_x1 - int((w_window/2.0) - self.maxLeft[0][0])) > 0:
                left_x1 -= int((w_window/2.0) - self.maxLeft[0][0])
                left_x2 -= int((w_window/2.0) - self.maxLeft[0][0])
                left_y1 -= int(h_window)
                left_y2 -= int(h_window)
                
            else:
                left_y1 -= int(h_window)
                left_y2 -= int(h_window)

            if self.maxRight[0].size != 0 and (right_x1 - int((w_window/2.0) - self.maxRight[0][0])) > 0:
                right_x1 -= int((w_window/2.0) - self.maxRight[0][0])
                right_x2 -= int((w_window/2.0) - self.maxRight[0][0])
                right_y1 -= int(h_window)
                right_y2 -= int(h_window)
            else:
                right_y1 -= int(h_window)
                right_y2 -= int(h_window)
            """
            #left_lane_coords_x = np.append(left_lane_coords_x, left_x1 + (w_window/2.0))
            #right_lane_coords_x = np.append(right_lane_coords_x, right_x1 + (w_window/2.0))

            #lane_coords_y = np.append(lane_coords_y, left_y1 + h_window)



            #cv.imshow("Left", left_window)
            #cv.imshow("Right", right_window)

            if cv.waitKey(0) == 27:
                continue

            #graph lines on image
            #x = right_lane_coords_x
            #y = np.polyfit(x, lane_coords_y, 2)
            #p = np.poly1d(y)
            #plt.plot(x,p(x))
            #plt.show()
            #print(p(x))
            #print(x)
            #draw_points = (np.asarray([x, p(x)]).T).astype(np.int32)
            #self.og_frame = cv.polylines(self.og_frame,[draw_points], isClosed=False,color=(0,255,0),thickness=5)
            

            #except Exception as e:
            #    print("Error: Determining inital lane positons\n" + str(e))
                #TODO: as a result nothing should happen and next fraame should be read and redo sliding windows

    def to_binary(self):
        ret3,self.frame = cv.threshold(self.frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    def blur(self):
        self.frame = cv.GaussianBlur(self.frame, ksize=(5,5), sigmaX=0,sigmaY=0)
    
    def mask_lanelines(self):
        """
        Masks the yellow and white lane lines and used sobel filter 
        -takes in color image

        TODO: test other color spaces
        White pixel detection: R-channel (RGB) and L-channel (HLS)
        Yellow pixel detection: B-channel (LAB) and S-channel (HLS)
        """
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
        #self.frame = self.temp_img

        #sobel filters
        x_out = cv.CV_16S
        y_out = cv.CV_16S
        scale = 1
        delta = 0
        kernal = np.ones((5,5),np.uint8)
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray, x_out, 1, 0, ksize=self.sobelX, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) #size was 3
        grad_y = cv.Sobel(gray, y_out, 0, 1, ksize=self.sobelY, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT) #size was 3

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        
        #TODO: add slider to tweak values
        #combine sobel x and y
        self.frame = cv.addWeighted(abs_grad_x, 2, abs_grad_y, 1, 0)
        #erosion and dilation of Sobel image
        self.frame = cv.morphologyEx(self.frame, cv.MORPH_OPEN, kernal)

        #TODO: use bitwise, not weighted image also tweak weights; test different ones to see best one
        #self.frame = cv.addWeighted(self.frame, 1, self.temp_img, 1, 0)
        #self.frame = cv.bitwise_or(self.frame, self.temp_img)
        
        #joins HSL color threshold and Sobel image TODO: test also suing canny edge detection
        self.frame = cv.add(self.frame,self.temp_img)        

    def mask_lanes_alt(self):
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

        #eroding iamge  NOTE: Erosion does a lot more hurting than helping
        self.frame = cv.erode(self.frame,kernel, iterations=1)

        #adding canny/sobel with color threshold
        self.frame = cv.add(self.frame, self.temp_img)

    def transform(self):
        #TODO: Fix point order portion currently not using
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        self.perspectivePointsOrdered = np.zeros((4, 2), dtype = "float32")
        #print(self.perspectivePointsOrdered)
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = np.sum(self.perspectivePoints, axis = 1)
        #print(s)
        self.perspectivePointsOrdered[0] = self.perspectivePoints[np.argmin(s)]
        self.perspectivePointsOrdered[2] = self.perspectivePoints[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the alargest difference
        diff = np.diff(self.perspectivePoints, axis = 1)
        self.perspectivePointsOrdered[1] = self.perspectivePoints[np.argmin(diff)]
        self.perspectivePointsOrdered[3] = self.perspectivePoints[np.argmax(diff)]

        #print(self.perspectivePoints)
        #print(self.perspectivePointsOrdered)
        # obtain a consistent order of the points and unpack them
	    # individually
        (tl, tr, br, bl) = self.perspectivePoints

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv.getPerspectiveTransform(self.perspectivePoints,dst)
        self.frame = cv.warpPerspective(self.frame, M, (maxWidth, maxHeight), flags=cv.INTER_LINEAR)

    def object_rec(self):
        """
        frame should be gray and smaller the better run time
        """
        #gets coordinates of faces TODO: test diffretne scale facotrss 1.0485258 was using 1.1 and 7
        self.gray = cv.cvtColor(self.og_frame, cv.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(self.gray, 1.1, 5)
        if len(objects) > 0:
            #reset faces if there are new faces
            self.object = []
        #draw faces
        for (x, y, w, h) in objects:
            cv.rectangle(self.og_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(self.og_frame, 'Car', (x + 6, y - 6),fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)

            #if faces.size > 0:
            #[left_y1:left_y2, left_x1:left_x2]
            self.objects.append(self.og_frame[y:y+h, x:x+w])


if __name__ == "__main__":
    #test1 = Video("test_videos/my_captures/trim11_45_03_2020-05-11.mp4",fps=24)
    #test1.display()
    #test1.transform()
    #test1.histogram()

    #test2 = Video("test_videos/my_captures/trim11_45_03_2020-05-11.mp4")
    #test2 = Video("test_videos/my_captures/trim11_51_08_2020-05-11.mp4")
    test2 = Video('test_videos/my_captures/capture11_56_19_2020-05-11.mp4')
    #test2 = Video("test_videos/my_captures/trim11_41_01_2020-05-11.mp4")
    #test2 = Video("/Volumes/Main-Drive/AutonoMas/Trim_17__05-25.mp4")
    test2.display()

    #test3 = Video(1,name='Real Time 1')
    #test3.display()