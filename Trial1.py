import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the range of colors for the crosshair
lower_gray = np.array([0, 0, 0])
upper_gray = np.array([150, 150, 150])

# Define the size of the crosshair
crosshair_size = 50

# Initialize variables to keep track of the maximum offset
max_offset_x = 0
max_offset_y = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the grayscale image to separate the crosshair from the background
        ret, thresh = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

        # Find the contours of the crosshair in the thresholded image
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Draw the contours on the original image
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

        # Iterate through each contour
        for contour in contours:

            # Find the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore contours that are too small to be the crosshair
            if w < crosshair_size or h < crosshair_size:
                continue

            # Calculate the center of the bounding rectangle
            cx = x + w // 2
            cy = y + h // 2

            # Calculate the offset from the center of the image
            offset_x = cx - frame.shape[1] // 2
            offset_y = cy - frame.shape[0] // 2

            # Update the maximum offset if necessary
            max_offset_x = max(max_offset_x, offset_x)
            max_offset_y = max(max_offset_y, offset_y)

            # Draw a circle at the center of the crosshair
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Display the image with the crosshair and the maximum offset
        cv2.putText(frame, "Max offset: ({}, {})".format(max_offset_x, max_offset_y), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
