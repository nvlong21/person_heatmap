import cv2
import scipy.io as sio
import os
from center_detect import center_detect
import time
import numpy as np
from scipy import ndimage, misc
import cv2
import numpy as np

ix,iy = -1, -1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        ix,iy = x,y

# Create a black image, a window and bind the function to window

cv2.destroyAllWindows()
if __name__ == '__main__':
    img = cv2.imread('heatmap.jpg')
    # img_45 = ndimage.rotate(img, 45, reshape=False)
    # cv2.imwrite('a.jpg', img_45)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    pts_src = []
    pts_dst = []
    i = 0
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            i+=1
            pts_src.append([ix,iy])
            if i==4:
                break
    print(pts_src)
    i = 0
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            i+=1
            pts_dst.append([ix,iy])
            if i==4:
                break
        # provide points from image 1
    pts_src = np.array(pts_src, dtype=np.float32)
    # # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))

    pts_dst = np.array(pts_dst, dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (512, 512))
    cv2.imwrite('aas.jpg',warped)


    # finally, get the mapping
    # pointsOut = cv2.perspectiveTransform(a, h)
