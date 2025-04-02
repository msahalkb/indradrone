import cv2 as cv
import numpy as np
ref = cv.imread('C:/Users/LENOVO/Downloads/drone_visual.jpg')
ref = cv.resize(ref, (int(1000 / 2), int(667 / 2)))
gaus = cv.GaussianBlur(ref, (5, 5), 2)
# Convert to HSV color space
hsvr = cv.cvtColor(ref, cv.COLOR_BGR2HSV)

# Define color ranges for trees (green), water (blue), and buildings (gray/white)
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_gray = np.array([0, 0, 200])
upper_gray = np.array([180, 50, 255])

# Create masks for each obstacle type
mask_green = cv.inRange(hsvr, lower_green, upper_green)
mask_blue = cv.inRange(hsvr, lower_blue, upper_blue)
mask_gray = cv.inRange(hsvr, lower_gray, upper_gray)

# Combine all masks
combined_mask = cv.bitwise_or(mask_green, mask_blue)
combined_mask = cv.bitwise_or(combined_mask, mask_gray)

# Generate a heatmap from the combined mask
heatmap = cv.applyColorMap(combined_mask, cv.COLORMAP_JET)

# Overlay the heatmap on the original image
overlay = cv.addWeighted(ref, 0.7, heatmap, 0.3, 0)

# Find contours for the detected obstacles
contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    area = cv.contourArea(contour)
    if area > 500:  # Filter small areas
        cv.drawContours(ref, [contour], -1, (0, 0, 255), 2)

# Display the results
cv.imshow("Original Image", ref)
cv.imshow("Obstacle Mask", combined_mask)
cv.imshow("Heatmap", heatmap)
cv.imshow("Overlay (Heatmap + Original)", overlay)
cv.waitKey(0)
cv.destroyAllWindows()