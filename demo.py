import cv2
import scipy.io as sio
import os
from center_detect import center_detect
import time
import numpy as np
# from connector.db import employee, history, room_db
# from connector.db_conector import *
# from connector.base import Base, Session, engine
from scipy import ndimage, misc
# Base.metadata.create_all(engine)
ix,iy = -1, -1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print("aaaaaa")
        # cv2.circle(img,(x,y),5,(255,0,0),-1)
        ix, iy = x,y
        print(x, y)

def camera():

    cap = cv2.VideoCapture("TownCentreXVID.mp4")

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    
    pts_src = []
    pts_dst = []
    i = 0
    tranform = False
    if tranform:
        img = cv2.resize(frame.copy(), (768, 768))
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)
        
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
        print(pts_src, pts_dst)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    detector = center_detect([w, h], [w, h])
    out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, (768, 768))

    hm = None
    while True:
        t = time.time()
        ret, frame = cap.read()
        if ret:
            t = time.time()
            dets, heatmap = detector(frame, threshold=0.35)
            heatmap = np.where(heatmap <= 0.35, 0, 1)
            if hm is not None:
                hm += heatmap.copy()
            else:
                hm = heatmap.copy()
            heatmap = detector.gen_heatmap(hm.copy())
            result = cv2.addWeighted(frame, 0.5,cv2.resize(heatmap, (w, h)),0.3,0)

            for det in dets:
                boxes, score = det[:4], det[4]
                cv2.rectangle(result, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            if tranform:
                result = cv2.warpPerspective(cv2.resize(result, (768, 768)), M, (768, 768))
            out.write(result)
            cv2.imshow('out', result)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cam_heatmap = detector.gen_heatmap(hm)
    if tranform:
        cam_heatmap = cv2.warpPerspective(cv2.resize(cam_heatmap, (768, 768)), M, (768, 768))
    cam_heatmap = cv2.resize(cam_heatmap, (w, h))
    cam_heatmap = cv2.addWeighted(frame, 0.3, cam_heatmap,0.5, 0)
    cv2.imwrite("heatmap.jpg", cam_heatmap)
    cap.release()
    out.release()

def test_image():
    frame = cv2.imread('imgs/s3.jpg')
    # frame = cv2.resize(frame, (512, 512))
    h, w = frame.shape[:2]
    centerface = center_detect(h, w)
    t = time.time()        
    dets, heatmap = centerface(frame, threshold=0.25)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    cv2.imwrite('centerface.jpg', frame)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


def test_widerface():
    Path = 'widerface/WIDER_val/images/'
    wider_face_mat = sio.loadmat('widerface/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = 'save_out/'

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        # print(save_path + im_dir)
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img = cv2.imread(os.path.join(Path, zip_name))
            h, w = img.shape[:2]
            landmarks = True
            centerface = CenterFace(h, w, landmarks=landmarks)
            if landmarks:
                dets, lms = centerface(img, threshold=0.05)
            else:
                dets = centerface(img, threshold=0.05)
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets:
                x1, y1, x2, y2, s = b
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    camera()
    # test_image()
    # img = cv2.imread('heatmap.jpg')
    # img_45 = ndimage.rotate(img, 45, reshape=False)
    # cv2.imwrite('a.jpg', img_45)
