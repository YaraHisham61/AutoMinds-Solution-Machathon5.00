#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
import matplotlib.pyplot as plt
import numpy as np
import time

class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

class SolutionNode(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        ### Subscriber to the image topic
        self.subscriber = self.create_subscription(Image,"/prius/front_camera/image_raw",self.callback,10)
        ### Publisher to the control topic
        self.publisher = self.create_publisher(Control, "/prius/control", qos_profile=10)
        self.fps_counter = FPSCounter()
        self.bridge = CvBridge()
        self.command = Control()
        # while(1):
        #     self.command.throttle = 1.0
        #     self.command.shift_gears = Control.FORWARD
        #     self.command.steer = 0.0
        #     self.publisher.publish(self.command)
    
    def draw_fps(self, img):
        self.fps_counter.step()
        fps = self.fps_counter.get_fps()
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img

    def canny(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def display_lines(self,img,lines):
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        return line_image

    def region_of_interest(self,canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)

        polygons = np.array([[
        (0, 300),
        (0, height),
        (width, height),
        (width, 300)]], np.int32)

        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image
    
    def make_points(self,image, line):
        slope, intercept = line
        slope += 0.001
        y1 = int(image.shape[0])# bottom of the image
        y2 = int(y1*3/5)         # slightly lower than the middle
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y = int(image.shape[0]*3/4)        
        x = int((y - intercept)/slope)
        return [[x1, y1, x2, y2]],x,y
    def line_select(self,image, lines):
        left_fit    = []
        left_line = []
        right_fit   = []
        right_line = []
        if lines is None:
            return [] , (image.shape[0]//2,int(image.shape[1]*3/4))        
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0: # y is reversed in image
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        # add more weight to longer lines
        # check if left_fit is empty
        if not left_fit:
            # something
            x1 = 0
            y1 =  300
        else:
            left_fit_selected  = np.average(left_fit, axis=0)
            left_line,x1,y1  = self.make_points(image, left_fit_selected)

        if not right_fit:
            x1 = image.shape[1]
            y1 =  300
        else:
            right_fit_selected = np.average(right_fit, axis=0)
            right_line,x2,y2 = self.make_points(image, right_fit_selected)
        
        midpoint= ((x1+x2)//2,(y1+y2)//2)
        selected_lines = [left_line, right_line]
        return selected_lines,midpoint

    def callback(self,msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = self.draw_fps(cv_image)
        canny = self.canny(cv_image)
        cropped_canny = self.region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        selected_lines,current_point = self.line_select(cv_image,lines)
        target_point = (cv_image.shape[1]//2,current_point[1])
        if selected_lines != None:
            line_image = self.display_lines(cv_image,selected_lines)
            line_image = cv2.circle(line_image,current_point, radius=1, color=(0, 0, 255), thickness=-1)
            line_image = cv2.circle(line_image,target_point, radius=1, color=(0, 255, 0), thickness=-1)
            cv2.imshow("line_image",line_image)      

        #try to find the target point
        # print("target_point",target_point)
        # print("current_point",current_point)
        # print("distance",math.sqrt((target_point[0]-current_point[0])*2+(target_point[1]-current_point[1])*2))
        


        if (target_point[0] - current_point[0]) >= 20:
            # go right
            print("go right")
            self.command.throttle = 0.1
            self.command.steer = 1.0
        elif (target_point[0] - current_point[0]) <= -20:
            # go left
            print("go left")
            self.command.throttle = 0.1
            self.command.steer = -1.0
        else:
            # go straight
            print("go straight")
            self.command.throttle = 0.5
            self.command.steer = 0.0
        self.publisher.publish(self.command)


        #### show image
        # cv2.imshow("ROI",self.region_of_interest(canny))
        # cv2.imshow("canny",canny)
        cv2.imshow("prius_front",cv_image)  
        cv2.waitKey(5)



def main():
    rclpy.init()
    node = SolutionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()