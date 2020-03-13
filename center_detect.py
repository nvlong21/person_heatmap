import numpy as np
import cv2
import datetime
from model.centernet import EfficientNet
import torch
from collections import OrderedDict
from torchvision import transforms as trans
import time


class center_detect(object):
    mean = np.array([0.485, 0.456, 0.406],
                                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225],
                                     dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, input_size = [640, 640], output_size = [640, 640]):
        self.net = EfficientNet()
        self.cuda = True
        if self.cuda:
            self.net.cuda()
        checkpoint = torch.load('weights/model_epoch_95.pt')
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        del checkpoint
        self.height = input_size[1]
        self.width = input_size[0]
        self.output_size = output_size
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(self.height, self.width)
        print(self.img_h_new, self.img_w_new)
    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def __call__(self, img, threshold=0.5):
        height, width = img.shape[0:2]

        img = self.process_input(img)
        if self.cuda:
            img = img.cuda()
        out = self.net(img)[0]
        heatmap, scale, offset = torch.clamp(out['hm'].sigmoid_(), min=1e-4, max=1-1e-4).detach().cpu().numpy(), out['wh'].detach().cpu().numpy(),\
             out['reg'].detach().cpu().numpy()
        dets = self.decode(heatmap.copy(), scale, offset, (self.height, self.width), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = (dets[:, 0:4:2]/self.img_w_new*self.output_size[0])//1, (dets[:, 1:4:2]/ self.img_h_new*self.output_size[1])//1 #// self.scale_w, self.scale_h 
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
        return dets, heatmap

    def process_input(self, img):

        img = cv2.resize(img, (self.img_w_new, self.img_h_new))
        img = (img.astype(np.float32) / 255.)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img)
        img = torch.unsqueeze(img, 0)
        return img

    def gen_heatmap(self, hm):
        hm = np.squeeze(hm)
        cam = hm - np.min(hm)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        return heatmap

    def decode(self, heatmap, scale, offset, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        print(np.max(heatmap))
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = scale0[c0[i], c1[i]]*4, scale1[c0[i], c1[i]]*4
                # s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s0 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s1 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s0, size[1]), min(y1 + s1, size[0]), s])
               
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
        return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep
