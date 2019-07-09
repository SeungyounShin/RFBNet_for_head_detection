import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from skimage.transform import resize

import sys

import cv2

from .transform import TestBaseTransform



class ImageFolderCombine(Dataset):

    def __init__(self, folder_path, yolo_img_size, dsfd_shrink):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.yolo_img_shape = (yolo_img_size, yolo_img_size)
        self.dsfd_shrink = dsfd_shrink
        self.dsfd_transform = TestBaseTransform((104, 117, 123))


    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]

        ## Get img
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        ## Generate img_yolo
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        img_yolo = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        img_yolo = resize(img_yolo, (*self.yolo_img_shape, 3), mode='reflect')
        # Channels-first
        img_yolo = np.transpose(img_yolo, (2, 0, 1))
        # As pytorch tensor
        img_yolo = torch.from_numpy(img_yolo).float()

        ## Generate img_dsfd
        if self.dsfd_shrink != 1:
            img_dsfd = cv2.resize(img, None, None, fx=self.dsfd_shrink, fy=self.dsfd_shrink, interpolation=cv2.INTER_LINEAR)
        img_dsfd = self.dsfd_transform(img_dsfd)[0]
        # As pytorch tensor
        img_dsfd = torch.from_numpy(img_dsfd)
        img_dsfd = img_dsfd.permute(2, 0, 1)

        return img_yolo, img_dsfd


    def __len__(self):
        return len(self.files)



class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


class ImageFolderFace(Dataset):

    def __init__(self, folder_path, shrink):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.shrink = shrink
        self.transform = TestBaseTransform((104, 117, 123))


    def __getitem__(self, index):
        # Extract image
        img_path = self.files[index % len(self.files)]
        img = cv2.imread(img_path)
        img_og = img

        if self.shrink != 1:
            img = cv2.resize(img, None, None, fx=self.shrink, fy=self.shrink, interpolation=cv2.INTER_LINEAR)

        img = self.transform(img)[0]

        # As pytorch tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img_og, img


    def __len__(self):
        return len(self.files)
