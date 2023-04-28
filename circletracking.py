import numpy as np
import cv2

cap = cv2.VideoCapture(0)

r = [290,560,5000,200]

lower = np.array([0,0,0])
upper = np.array([255,255,255])

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:

        rgb_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(frame, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        img_erode =  cv2.dilate(mask,kernel, iterations=10)
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = []

        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
        drawing = np.zeros((img_erode.shape[0], img_erode.shape[1], 3), np.uint8)

        for i in range(len(hull)):
            M = cv2.moments(hull[i])

            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(drawing, (cX, cY), 5, (15, 15, 25), -1)

            except Exception as e:
                print(e)
                pass
    
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
