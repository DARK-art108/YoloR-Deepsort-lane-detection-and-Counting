import cv2
import numpy as np

def get_lanes(source):

    first_frame = None
    roi = None
    video = cv2.VideoCapture(source)
    showCrosshair = False
    fromCenter = False
    lane_lines = None
    line_for_intersection = None

    # read only first frame and extract lines
    if first_frame is None:

        _, first_frame = video.read()

        # use keyboard and get ROI area
        roi = cv2.selectROI("Select area", first_frame, fromCenter, showCrosshair)

        # crop ROI
        imCrop = first_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        # Perform dilation to increase size of white color  
        kernel = np.ones((3,3),np.uint8)
        first_frame = cv2.dilate(imCrop,kernel,iterations = 1) 

        # convert to gray scale 
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        ######################## change 200 based on lighting conditions
        # Threshold the gray image to get only white colors
        white_lane_mask = cv2.inRange(gray, 200, 255)
        ########################

        # bit wise and and generate mask speedy way (without for loop)
        new_masked = cv2.bitwise_and(gray, gray, mask= white_lane_mask)
        cv2.imshow("new_masked", new_masked)
        
        # apply threshold
        thresh, gray = cv2.threshold(new_masked,150,255,cv2.THRESH_BINARY)
        cv2.imshow("AO1", gray)

        # find lane lines using edge detection
        edges = cv2.Canny(gray, 0.3*thresh, thresh)
        lane_lines = cv2.HoughLinesP(edges,2,np.pi/180,30,minLineLength=15,maxLineGap=40)
        line_for_intersection = cv2.HoughLines(edges,1,np.pi/180,50)
        cv2.imshow("edges", edges)
        cv2.waitKey(0)

    # return lines and ROI
    return lane_lines, line_for_intersection, roi

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_intersection_points(lines, horizontal_line, r):

    intersection_points = []

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b)) + r[0]
        y1 = int(y0 + 1000*(a))+ r[1]
        x2 = int(x0 - 1000*(-b)) + r[0]
        y2 = int(y0 - 1000*(a)) + r[1]

        # find intersection points
        intersection_point = line_intersection(horizontal_line, [(x1, y1), (x2 , y2)])

        if horizontal_line[1][0] >= intersection_point[0] and horizontal_line[0][0] <= intersection_point[0]:
            intersection_points.append(intersection_point)

    return intersection_points

def draw_intersection(points):
    for pts in points:
        cv2.circle(frame, (int(pts[0]),int(pts[1])), 3,(0,255,255),3)

############################### select lane_size in pixel based on video
# partition of horizontal_line
def get_small_lines(intersection_points, lane_size = 60):
    small_lines = []
    x_values = []
    start_point = None

    for pts in intersection_points:
        x_values.append(int(pts[0]))

    # print("x_values", sorted(x_values))

    for xy_pts in sorted(x_values):
        if start_point is None:
            start_point = xy_pts
        else:
            ############################### lane size in pixel
            if abs(xy_pts - start_point) >= lane_size:
                small_lines.append([(start_point + 15 ,500), (xy_pts -15, 500)])        
                start_point = xy_pts
            else:
                pass

    # print("small_lines", small_lines)

    return small_lines
    
# This function is only for unit test of above function 
if __name__ == '__main__' :

    video_path = "../720.mp4"
    
    video = cv2.VideoCapture(video_path)    

    lines, line_for_intersection, r = get_lanes(video_path)

    ############################## change 300 and 1060 based on road size
    line1 = [(300,500), (1060, 500)]
    ##############################

    intersection_points = get_intersection_points(line_for_intersection, line1, r)

    # print(intersection_points)
    small_lines = get_small_lines(intersection_points)
    
    # print(small_lines)
    # small_lines.pop(2)

    while True:
        
        check, frame = video.read()
        if check == False:
            break

        # draw line 
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1+ r[0], y1 + r[1]), (x2 + r[0], y2 + r[1]), (255, 0, 0), 3)

            # draw_intersection(intersection_points)

        for li in small_lines:
            cv2.line(frame, li[0], li[1], (0,200,0), 3)        
            cv2.putText(frame, str(0), (li[0][0]+50, li[0][1] + 30), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        
        # Display output
        cv2.imshow("Demo", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
