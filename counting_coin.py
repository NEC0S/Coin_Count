import numpy as np
import cv2
import cvzone

# Load the images and resize it
img = cv2.imread("Image111.jpg")
img = cv2.resize(img, (640, 800))
image_copy = cv2.imread("Image1.jpg")
image_copy = cv2.resize(image_copy, (640, 800))


# Function for the trackbar callback
def empty(a):
    pass

# Create a window for trackbars
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)

# Create trackbars for adjusting thresholds
cv2.createTrackbar("Threshold1", "Settings", 39, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 8, 255, empty)


# Creat a Preprocessing function to apply necessary operations on the input image
def preProcessing(img):
    
    # Convert image to grayscale
    imgPre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to the image
    imgPre = cv2.blur(img, (15, 15), 0)
    

    # Get threshold values from trackbars
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    
    # Apply Canny edge detection
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    # Apply dilation and morphological closing
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre

# Call preprocessing function
imgPre = preProcessing(img)

# Find contours in the preprocessed image
imgContours, conFound = cvzone.findContours(img, imgPre, minArea=2000)

    
# Stack the images for display
imgStacked = cvzone.stackImages([image_copy, img, imgPre, imgContours], 2, 0.5)

# Create a dictionary to store contour areas
area = {}

# Calculate and store the area of each contour
for i in range(len(conFound)):
    cnt = conFound[i]
    ar = cv2.contourArea(cnt['cnt'], True)
    area[i] = np.abs(ar)
    print(area)
# Sort the areas in descending order
srt = sorted(area.items(), key=lambda x: x[1], reverse=True)

# Convert the sorted results to a numpy array
results = np.array(srt).astype("int")

# Count the number of contours with area less than -2000
num = np.argwhere((results[:, 1]) > 2000).shape[0]
print("Number of coins is ", num)

# Display the stacked images
cv2.imshow("Image", imgStacked)
cv2.waitKey(0)



    
