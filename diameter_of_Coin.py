import cv2
import numpy as np

# Load the image
image = cv2.imread('fordia.jpg')
image = cv2.resize(image, (640, 558))
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.medianBlur(gray,5)


# Use HoughCircles to detect circles in the image
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=35,
    param1=70,
    param2=48,
    minRadius=50,
    maxRadius=100
)

# If circles are found, draw them and calculate diameter
if circles is not None:
    circles = np.uint16(np.around(circles))
    
    for i, circle in enumerate(circles[0, :]):
        center = (circle[0], circle[1])
        radius = circle[2]
        
        # Draw the circle
        cv2.circle(image, center, radius, (255, 255, 0), 3)

        # Calculate diameter
        diameter = 2 * radius
        #for 680 pixels we have 10cm so
        diameter=diameter*(10/680)
        # print(f"Diameter of circle: {diameter} cm")
        print(f"Diameter of circle {i + 1}: {diameter:.2f} centimeters")
    # Display the result
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
