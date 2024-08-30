import os
import torch
import pandas as pd
import numpy as np

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms as T
import cv2


class BoneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, validation=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))
        self.annotations = os.path.join('./data/landmarksBB.csv')
        self.validation = validation

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = cv2.imread(img_path)

        ann_df = pd.read_csv(self.annotations)

        box = ann_df[ann_df['image_name'].str.contains(self.imgs[idx])].iloc[0, 23:]
        boxes = [[box.iloc[0], box.iloc[1], (box.iloc[0] + box.iloc[2]), (box.iloc[1] + box.iloc[3])]]

        keypoints_flat = ann_df[ann_df['image_name'].str.contains(self.imgs[idx])].iloc[0, 1:23].to_numpy(dtype='float32', copy=True)
        keypoints = np.reshape(keypoints_flat, (-1, 2)).tolist() 

        #keypoints = [[[keypoint[0], keypoint[1], 1] for keypoint in keypoints]]

        if self.transforms:
            keypoints = [(keypoint[0], keypoint[1]) for keypoint in keypoints]
            augmented = self.transforms(image=img, keypoints=keypoints, bboxes=boxes, category_ids=[0])
            img = T.ToTensor()(augmented['image'])
            keypoints = augmented['keypoints']
            boxes = augmented['bboxes']

        keypoints = [[[keypoint[0], keypoint[1], 1] for keypoint in keypoints]]

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["keypoints"] = torch.FloatTensor(keypoints)
        target["labels"] = torch.ones((1), dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = box.iloc[2] * box.iloc[3]
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)

        return [img, target]
    
    def __len__(self):
        return len(self.imgs)

