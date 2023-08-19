from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import torchstain
from typing import Dict, Literal

class PatchDataset(Dataset):
    def __init__(
            self, img_paths, mask_paths, mode: str,
            name_to_class_mapping: Dict,
            stain_normaliser: torchstain.normalizers.HENormalizer,
            level: Literal["patch", "pixel"] = "patch",
            patch_area_threshold: int = 0.4,
            white_threshold: int = 230,
            transform=None, seed=0
        ):
        self.imgs = img_paths
        self.masks = mask_paths
        self.transform = transform
        self.mode = mode
        self.name_to_class_mapping = name_to_class_mapping
        self.stain_normaliser = stain_normaliser
        self.level = level
        self.patch_area_threshold = patch_area_threshold
        self.white_threshold = white_threshold
        self.seed = seed
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(f"image path: {img_path}, mask path: {mask_path}")

        # white thresholding
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_majority_white = bw_img.mean() > self.white_threshold

        if not is_majority_white:
            try:
                img, _, _ = self.stain_normaliser.normalize(img)
            except Exception as e:
                print(f"Error in normalising image: {img_path}")
                print(e)
            else:
                mask = np.zeros_like(mask)

        if self.mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.mode == "CIELAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif self.mode == "BW":
            img = bw_img

        class_id = self.name_to_class_mapping["-".join(img_path.split("/")[-3].split("-")[:-2])]

        # set seed so that the same transformation is applied to image and mask
        if self.transform:
            # transform the image
            random.seed(self.seed)
            ground_truth = None
            if self.level == "pixel": # for segmentation
                transformed = self.transform(image=img, mask=mask)
                img, mask = transformed["image"].float()/255.0, transformed["mask"]/255.0
                mask[mask > 0.5] = class_id
                mask[mask <= 0.5] = 0
                mask = mask.long()
                ground_truth = mask
            elif self.level == "patch": # for patch level classification
                transformed = self.transform(image=img)
                img = transformed["image"].float()/255.0
                if mask.mean() >= self.patch_area_threshold:
                    ground_truth = class_id
                else:
                    ground_truth = 0
            else:
                raise Exception("Invalid level")
            
            self.seed += 1
            
        return img, ground_truth
    
class SlideROIDataset(Dataset):
    def __init__(
            self, img_paths, roi_paths,
            transform=None, seed=0
        ):
        self.imgs = img_paths
        self.rois = roi_paths
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        roi_path = self.rois[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rois = np.load(roi_path).reshape(-1, 4)
        class_labels = np.ones(rois.shape[0])
        # print(f"image path: {img_path}, roi path: {roi_path}")
        
        # set seed so that the same transformation is applied to image and mask
        if self.transform:
            # transform the image
            random.seed(self.seed)

            transformed = self.transform(image=img, bboxes=rois, class_labels=class_labels)
            img = transformed["image"].float()/255.0
            rois = transformed["bboxes"]
            class_labels = transformed["class_labels"]
            
            self.seed += 1
            
        return img, rois, class_labels