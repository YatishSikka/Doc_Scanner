import numpy as np
import matplotlib.pyplot as plt
import cv2

im_path = "image.jpg"
img = cv2.imread(im_path)

img = cv2.resize(img, (1000,800))

#image blurring
orig = img.copy()
gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray,(5,5),0)

regen = cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)

#edge detection
edge = cv2.Canny(blurred,0,50)
orig_edge = edge.copy()

#contour extraction

contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
countours = sorted(contours, reverse=True, key = cv2.contourArea)

for c in contours:
    p = cv2.arcLength(c,True)

    approx = cv2.approxPolyDP(c, 0.01*p, True)

    if len(approx)==4:
        target = approx
        break

#reorder target contour

def reorder(h):

    h = h.reshape((4,2))

    hnew = np.zeros((4,2), dtype=np.float32)

    add = h.sum(axis=1)
    hnew[3] = h[np.argmax(add)]
    hnew[1] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[0] = h[np.argmax(diff)]
    hnew[2] = h[np.argmax(diff)]

    return hnew

reordered = reorder(target)
input_repr = reordered

output_map = np.float32([[0,0],[800,0],[800,800],[0,800]])

M = cv2.getPerspectiveTransform(input_repr,output_map)

ans = cv2.warpPerspective(orig,M,(800,800))


res = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)

b_res = cv2.GaussianBlur(res,(3,3),0)

plt.imshow(b_res)
plt.show()