import cv2
import numpy as np

from collections import deque


STARTING_FRAME = 200

cap = cv2.VideoCapture("videos/clair-de-lune.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
cap.set(cv2.CAP_PROP_POS_FRAMES, STARTING_FRAME)

line_buffer = deque()

while True:
    # Convert to grayscale
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Clip frame by removing top 300 pixels
    img = img[350:, :]

    # --- Apply CLAHE to improve local contrast ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Optionally save to visualize the contrast enhancement
    # cv2.imwrite("clahe_output.png", img_clahe)

    # --- Apply Gaussian Blur to reduce small noise ---
    blurred = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # --- Compute Sobel gradients ---
    sobelx = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    # cv2.imwrite("edges_gradient_output.png", gradient_magnitude)

    # --- Binary threshold to isolate strong edges ---
    _, edges = cv2.threshold(gradient_magnitude, 150, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("edges_output.png", edges)

    # --- Get coordinates of edge pixels ---
    ys, xs = np.where(edges > 0)
    height = img.shape[0]

    # --- Detect lines with probabilistic Hough Transform ---
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=20,
        maxLineGap=10,
    )

    # --- Filter for vertical lines ---
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan((y2 - y1) / (x2 - x1 + 1e-6)) * (180.0 / np.pi)
            if abs(angle) > 75:  # near-vertical
                vertical_lines.append((x1, y1, x2, y2))

    frame_x_centers = sorted(
        [int((x1 + x2) / 2) for (x1, y1, x2, y2) in vertical_lines]
    )

    merge_thresh = 6  # pixels
    # merged_xs = []

    # cluster = []
    # for x in frame_x_centers:
    #     if not cluster or abs(x - cluster[-1]) <= merge_thresh:
    #         cluster.append(x)
    #     else:
    #         merged_xs.append(int(np.mean(cluster)))
    #         cluster = [x]

    # merged_xs.append(int(np.mean(cluster)))
    # line_buffer.append(merged_xs)
    line_buffer.append(frame_x_centers)

    # Cluster and merge over multiple frames
    if len(line_buffer) > 20:
        all_clusters = []
        for frame_xs in line_buffer:
            all_clusters.extend(frame_xs)
        all_clusters = sorted(all_clusters)

        final_merged_xs = []
        cluster = []
        for x in all_clusters:
            if not cluster or abs(x - cluster[-1]) <= merge_thresh:
                cluster.append(x)
            else:
                final_merged_xs.append(int(np.mean(cluster)))
                cluster = [x]

        final_merged_xs.append(int(np.mean(cluster)))

        output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x in final_merged_xs:
            cv2.line(output_img, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)

        cv2.imwrite("lines_detected.png", output_img)
        exit(0)
        # Push back to the line buffer

    # --- Optional visualization ---
    # output_img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    # for x1, y1, x2, y2 in vertical_lines:
    #     cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imwrite("lines_detected.png", output_img)

    # output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # for x1, y1, x2, y2 in vertical_lines:
    #     cv2.line(
    #         output_img,
    #         (x1, y1),
    #         (x2, y2),
    #         (0, 0, 255),
    #         2,
    #     )
