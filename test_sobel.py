import cv2
import numpy as np

# Load image in grayscale
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Horizontal edges
sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Vertical edges

gradient_magnitude = cv2.magnitude(sobelx, sobely)

# Convert to uint8
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)


cv2.imwrite("edges_gradient_output.png", gradient_magnitude)


# Threshold to get binary edges (you can tweak 50)
_, edges = cv2.threshold(gradient_magnitude, 150, 255, cv2.THRESH_BINARY)


# Get edge pixel coordinates (y, x)
ys, xs = np.where(edges > 0)

height = img.shape[0]

# Save edge visualization
cv2.imwrite("edges_output.png", edges)

lines = cv2.HoughLinesP(
    edges,
    rho=1,  # distance resolution in pixels
    theta=np.pi / 180,  # angle resolution in radians
    threshold=8,  # minimum number of votes to consider a line
    minLineLength=20,  # minimum length of a line in pixels
    maxLineGap=18,  # maximum gap between line segments to link them
)

LINE_X_THRESHOLD = 20
LINE_Y_THRESHOLD = 5

# lines is an array of shape (num_lines, 1, 4), where each line is [x1, y1, x2, y2]
line_coords = []
vertical_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        m = (y2 - y1) / (x2 - x1 + 1e-6)  # slope
        line_coords.append((x1, y1, x2, y2))
        if abs(x2 - x1) < LINE_X_THRESHOLD and abs(y2 - y1) > LINE_Y_THRESHOLD:
            vertical_lines.append((x1, y1, x2, y2))

output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x1, y1, x2, y2 in vertical_lines:
    cv2.line(
        output_img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2,
    )

cv2.imwrite("lines_detected.png", output_img)
