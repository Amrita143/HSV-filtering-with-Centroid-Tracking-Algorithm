import cv2
import numpy as np
from centroidtracker import CentroidTracker

# read the video file and loop through the frames upto 3 second of the video file 
cap = cv2.VideoCapture('task_2_video.mp4')
# checking if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the tracker
ct = CentroidTracker(maxDisappeared=6)

# Define the range of green colors in HSV
lower_green = np.array([37, 63, 63])  # My Experiment --> [40,45%,30%] to [75,100%,100%]
upper_green = np.array([70, 255, 255])
fps = cap.get(cv2.CAP_PROP_FPS)
# count = 0
# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # # filter out the contours which are almost circular in shape
    # # Filter contours
    # filtered_cnts = []
    # min_area = 100  # Minimum area to be considered
    # circularity_threshold = 0.7  # Adjust this value based on your requirement

    # for cnt in cnts:
    #     area = cv2.contourArea(cnt)
    #     perimeter = cv2.arcLength(cnt, True)
    #     circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
        
    #     if area > min_area and circularity > circularity_threshold:
    #         filtered_cnts.append(cnt)

    # Calculate the center of each filtered contour and draw a red dot
    # for cnt in cnts:
    #     M = cv2.moments(cnt)
    #     if M["m00"] != 0:
    #         centerX = int(M["m10"] / M["m00"])
    #         centerY = int(M["m01"] / M["m00"])
            
    #         # Draw a red dot (circle) at the center
    #         cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)

    # Convert contours to the format expected by the tracker: (startX, startY, endX, endY)
    rects = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, x + w, y + h))
    
    # Update the tracker with the detected rectangles
    objects = ct.update(rects)
    
    # Draw red dots at the center of each tracked object
    for objectID, centroid in objects.items():
        cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 0, 255), -1)
    # Write the frame into the file 'output_video.mp4'
    out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()

# # HSV filtering to draw red dot at the Center of greeen balls in the image
# def resize_image(image, max_width=800):
#     # Calculate the ratio
#     h, w = image.shape[:2]
#     if w > max_width:
#         ratio = max_width / w
#         new_dim = (max_width, int(h * ratio))
#         resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
#         return resized
#     return image


# # read the image file and apply HSV filtering to draw red dot at the Center of greeen balls in the image
# frame = cv2.imread('frame_47.jpg')
# resized_img = resize_image(frame)
# cv2.imshow('resized_img', resized_img)
# cv2.waitKey(0)

# hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv)
# cv2.waitKey(0)
# # Define the range of green colors in HSV
# lower_green = np.array([36, 25, 25])  # My Experiment --> [40,45%,30%] to [75,100%,100%]
# upper_green = np.array([86, 255, 255])
# mask = cv2.inRange(hsv, lower_green, upper_green)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)

# cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# # Calculate the center of each filtered contour and draw a red dot
# for cnt in cnts:
#     M = cv2.moments(cnt)
#     if M["m00"] != 0:
#         centerX = int(M["m10"] / M["m00"])
#         centerY = int(M["m01"] / M["m00"])
        
#         # Draw a red dot (circle) at the center
#         cv2.circle(resized_img, (centerX, centerY), 5, (0, 0, 255), -1)


# cv2.imshow('frame', resized_img)
# cv2.waitKey(0)





