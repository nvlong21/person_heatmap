import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import random

class LoadImagesAndLabels(Dataset):  # for training
    mean = np.array([0.485, 0.456, 0.406],
                                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225],
                                     dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, root, path, img_size=[512,512], split = 'train'):
        with open(path, 'r') as file:
            self.imgs_path = file.readlines()
            self.imgs_path = [os.path.join(root, x).replace('\n', '') for x in self.imgs_path]
            self.imgs_path = list(filter(lambda x: len(x) > 0, self.imgs_path))

        random.shuffle(self.imgs_path)
        self.label_files = [x.replace('images', 'labels_with_ids').replace('Images', 'Annotations').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.imgs_path]
        self.nF = len(self.imgs_path)  # number of image files

        self.default_resolution = img_size
        self.not_rand_crop = False
        self.max_objs = 128
        self.keep_res = False
        self.down_ratio = 4
        self.mse_loss = False
        self.debug = 0
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                                         dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                                        dtype=np.float32)
        return bbox
 
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
                i *= 2
        return border // i

    def __getitem__(self, files_index):
        # print(self.imgs_path[files_index])
        if os.path.exists(self.imgs_path[files_index]):
            img = cv2.imread(self.imgs_path[files_index])
        else:
            print("%s not exists"%self.imgs_path[files_index])
                # Load labels
       
        height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.default_resolution[0], self.default_resolution[1]
        label_path = self.label_files[files_index]
        # print(label_path)
        if os.path.isfile(label_path):
            if 'WiderPerson' in label_path:
                list_bb = []
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue
                        lb, x1, y1, x2, y2 = line.replace('\n', '').split(' ')
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        list_bb.append(np.array(bbox))
                anns = np.array(list_bb)
                    # self.imgs_path = [os.path.join(root, x).replace('\n', '') for x in self.imgs_path]
                    # self.imgs_path = list(filter(lambda x: len(x) > 0, self.imgs_path))
    
            else:
                labels0_temp = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
                # Normalized xywh to pixel xyxy format
                labels0 = labels0_temp[:, 2:6].copy()
                anns = labels0.copy()
                anns[:, 0] = (labels0[:, 0] - labels0[:, 2] / 2)*width
                anns[:, 1] = (labels0[:, 1] - labels0[:, 3] / 2 )*height
                anns[:, 2] = (labels0[:, 0] + labels0[:, 2] / 2)*width
                anns[:, 3] = (labels0[:, 1] + labels0[:, 3] / 2)*height
        else:
            print(label_path)
            anns = np.array([])
        # print(anns)
        if self.debug :# and ("COCO" in str(self.imgs_path[files_index])):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # print(anns, label_path)
            plt.figure(figsize=(50, 50)) 
            plt.imshow(img)
            plt.plot(anns[:, [0, 2, 2, 0, 0]].T, anns[:, [1, 1, 3, 3, 1]].T, '.-')
            plt.axis('off')
            plt.savefig('%s'%self.imgs_path[files_index].split("/")[-1])
            time.sleep(10)
            # exit()

        num_objs = min(len(anns), self.max_objs)
        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1
        
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        num_classes = 1
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cls_id = 0
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.array(ann[:4].copy())
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                gt_det.append([4*(ct[0] - w / 2), 4*(ct[1] - h / 2), 
                           4*(ct[0] + w / 2), 4*(ct[1] + h / 2)])


        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

        if not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((1, 4), dtype=np.float32)
            meta = {'gt_det': gt_det}
            ret['meta'] = meta
        return ret

    def __len__(self):
        return self.nF  # number of batches


from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':
    data  =  LoadImagesAndLabels(root = "/media/hdd/sources/Person_2_heatmap/person", path = "/media/hdd/sources/Person_2_heatmap/person/data/full_path.train", img_size=[512, 512])
    dataloader_val = DataLoader(data, num_workers=4, batch_size=4,pin_memory=True,
            drop_last=False)
    for data in dataloader_val:
        pass