from torch.utils.data import Dataset
import cv2
import random
from typing import Dict

class PatchDataset(Dataset):
    def __init__(self, imgPaths, maskPaths, mode: str,
        name_to_class_mapping: Dict, transform=None, seed=0):
        self.imgs = imgPaths
        self.masks = maskPaths
        self.transform = transform
        self.mode = mode
        self.name_to_class_mapping = name_to_class_mapping
        self.seed = seed
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(f"image path: {img_path}, mask path: {mask_path}")
        
        if self.mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif self.mode == "CIELAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif self.mode == "BW":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        class_id = self.name_to_class_mapping["-".join(img_path.split("/")[-3].split("-")[:-2])]

        # set seed so that the same transformation is applied to image and mask
        if self.transform:
            # transform the image
            random.seed(self.seed)
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"].float()/255.0, transformed["mask"]/255.0
            mask[mask > 0.5] = class_id
            mask[mask <= 0.5] = 0
            mask = mask.long()
            self.seed += 1
            
        return img, mask