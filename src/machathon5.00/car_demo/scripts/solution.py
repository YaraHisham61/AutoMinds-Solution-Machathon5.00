#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
from sklearn.linear_model import RANSACRegressor
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt
import numpy as np
import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

        # Simulation parameters
        self.dt = 0.01         # Time step
        self.setpoint = setpoint    # Desired position


    def control(self, current_value):
        error = self.setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        # PID control equation
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error

        return output

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

        # Create PID controller
        pid_controller = PIDController(Kp=10, Ki=10, Kd=5,setpoint=427.0)

        self.pid_controller = pid_controller
        self.max_steering_angle = 1.0  # Maximum steering angle in degrees
        self.max_throttle = 0.5  # Maximum throttle (0 to 1)
        self.min_throttle = 0.35 # Minimum throttle during turns (0 to 1)

        # good numbers but very slow  
        
        # self.max_steering_angle = 0.8  # Maximum steering angle in degrees
        # self.max_throttle = 1.0  # Maximum throttle (0 to 1)
        # self.min_throttle = 0.35 # Minimum throttle during turns (0 to 1)
        self.right_turn = False
        self.left_turn = False  
        # self.command.throttle = 0.3
        # self.command.steer = 0.0
        self.publisher.publish(self.command)
    
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
    def draw_steer_throttle_control(self, img,steer,throttle,control):
        # Draw "steer" information
        cv2.putText(
            img,
            f"steer: {steer:.2f}",
            (10, 30),  # You may need to adjust the position based on your image
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Draw "throttle" information below the "steer" information
        cv2.putText(
            img,
            f"throttle: {throttle:.2f}",
            (10, 60),  # Adjust Y position to move it below the "steer" text
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
                # Draw "control" information
        cv2.putText(
            img,
            f"control: {control:.2f}",
            (10, 90),  # Adjust Y position to move it below the "throttle" text
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img


    def canny(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)
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
    
    def mask_red(self,img):
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0+mask1

        # set my output img to zero everywhere except my mask
        output_img = img.copy()
        # output_img[np.where(mask==0)]
        output_img = cv2.bitwise_and(output_img, output_img, mask= mask)

        return output_img


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
                
                
                # Use RANSACRegressor for robust fitting
                # ransac = RANSACRegressor()
                # ransac.fit(np.array([[x1, y1], [x2, y2]]), np.array([y1, y2]))
                # slope = ransac.estimator_.coef_[0]
                # intercept = ransac.estimator_.intercept_

                # p = P.fit(np.array([x1, x2]), np.array([y1, y2]),  1)
                # coefficients = p.convert().coef
                # intercept = coefficients[0]
                # slope = coefficients[1]

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
            self.right_turn = True
        else:
            self.right_turn = False
            left_fit_selected  = np.average(left_fit, axis=0)
            left_line,x1,y1  = self.make_points(image, left_fit_selected)

        if not right_fit:
            x1 = image.shape[1]
            y1 =  300
            self.left_turn = True
        else:
            self.left_turn = False
            right_fit_selected = np.average(right_fit, axis=0)
            right_line,x2,y2 = self.make_points(image, right_fit_selected)
       
        
        midpoint= ((x1+x2)//2,(y1+y2)//2)
        selected_lines = [left_line, right_line]
        return selected_lines,midpoint

    def callback(self,msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv_image = self.draw_fps(cv_image)
        mask_red = self.mask_red(cv_image)
        canny = self.canny(mask_red)
        cropped_canny = self.region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        selected_lines,current_point = self.line_select(cv_image,lines)
        target_point = (cv_image.shape[1]//2,current_point[1])
        if selected_lines != None:
            line_image = self.display_lines(cv_image,selected_lines)
            line_image = cv2.circle(line_image,current_point, radius=1, color=(0, 0, 255), thickness=-1)
            line_image = cv2.circle(line_image,target_point, radius=1, color=(0, 255, 0), thickness=-1)
            cv2.imshow("line_image",line_image)      
        control_output  = 0.0


        if self.left_turn == True:
            self.command.steer = -1.0
            self.command.throttle = 0.35
            # self.command.brake = 0.5
            self.left_turn = False
        elif self.right_turn == True:
            self.command.steer = 1.0
            self.command.throttle = 0.35
            # self.command.brake = 0.5
            self.right_turn = False
        else:            
            # Compute control output
            control_output = self.pid_controller.control(current_point[0])/10000.0
            
            # Apply control output
            # Steer control
            steering_adjustment = max(min(control_output, self.max_steering_angle), -self.max_steering_angle)
            self.command.steer = steering_adjustment
            # Throttle control
            # Reduce throttle based on the absolute value of the steering adjustment
            # if abs(steering_adjustment) > (self.max_steering_angle / 2):
            #     # Sharp turn, reduce speed
            #     throttle = self.min_throttle
            # else:
                # Mild steering, can accelerate
            throttle = self.max_throttle - (abs(steering_adjustment) / self.max_steering_angle) * (self.max_throttle - self.min_throttle)
            self.command.throttle = throttle
            # self.command.brake = 0.0
        # Print Steer and Throttle
        cv_image = self.draw_steer_throttle_control(cv_image,self.command.steer,self.command.throttle,control_output)
        # Publish control output
        self.publisher.publish(self.command)


        #### show image
        # cv2.imshow("ROI",self.region_of_interest(canny))
        # cv2.imshow("canny",canny)
        cv2.imshow("mask_red",mask_red)
        cv2.imshow("prius_front",cv_image)  
        cv2.waitKey(5)



def main():
    rclpy.init()
    node = SolutionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()