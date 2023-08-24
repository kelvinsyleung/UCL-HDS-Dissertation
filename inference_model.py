import os
from typing import List, Dict, Union, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt

from patchify import patchify
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

import cv2
import torch
import torch.nn as nn
import torchvision

from coord_utils import get_absolute_coordinates, pad_roi_coordinates
from log_utils import setup_logging
from class_mapping import LABELS2TYPE_MAP, LABELS2SUBTYPE_MAP
from patch_dataset import PatchDataset

OPENSLIDE_PATH  = r'C:/openslide/openslide-win64/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

class InferenceModel:
    def __init__(
        self, classifier_model: nn.Module, object_detection_model: nn.Module, device: str,
        slide_patch_size: int = 512, roi_patch_size: int = 512,
        white_threshold: float = 230,
        box_nms_threshold: float = 0.1,
        box_score_threshold: float = 0.1,
        box_area_threshold: float = 100,
        num_classes: int = 3
    ):
        self.object_detection_model = object_detection_model
        self.classifier_model = classifier_model
        self.slide_patch_size = slide_patch_size
        self.roi_patch_size = roi_patch_size
        self.white_threshold = white_threshold
        self.box_nms_threshold = box_nms_threshold
        self.box_score_threshold = box_score_threshold
        self.box_area_threshold = box_area_threshold
        self.device = device
        self.num_classes = num_classes
        setup_logging()
    
    def predict_proba(
        self,
        slide_filename: str
    ) -> List[Dict[str, Union[Tuple[int, int], np.ndarray]]]:
        """
        Inference on a slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to run inference on.
        
        Returns
        -------
            prob_list: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of dictionaries containing the coordinates of the patch and the probabilities of each class.

        Steps
        -----
        1. Extract patches from slide
        2. Flag slides that are majority white
        3. Runs object detection to get bounding boxes
        4. Extracts patches from bounding boxes
        5. Runs classifier on patches
        6. Aggregate predictions
        """
        slide = openslide.OpenSlide(slide_filename)

        # 1. Extract patches from slide
        slide_patches = self._extract_patches_from_slide(slide)
        logging.info(f"InferenceModel.predict - Extracted slide patches, shape: {slide_patches.shape}")

        # 2. Flag slides that are majority white
        majority_white = self._flag_majority_white(slide_patches)
        logging.info(f"InferenceModel.predict - Flagged majority white patches, proportion: {majority_white.mean()}")
        
        # 3. Runs object detection to get bounding boxes
        bbox_predictions = self._extract_bboxes(slide_patches, majority_white, bbox_predictions)
        logging.info(f"InferenceModel.predict - Extracted bounding boxes, number: {len(bbox_predictions)}")
        
        # 4. Extracts patches from bounding boxes
        # scale bounding boxes to absolute coordinates of the original image
        scaled_bboxes = self._scale_bboxes(slide, bbox_predictions)
        logging.info(f"InferenceModel.predict - Scaled bounding boxes, number: {len(scaled_bboxes)}")

        # extract patches from bounding boxes
        roi_patches = self._extract_patches_from_rois(slide, scaled_bboxes)
        logging.info(f"InferenceModel.predict - Extracted roi patches, number: {len(roi_patches)}")

        # 5. Runs classifier on patches
        patch_predictions = self._classify_patches(roi_patches)
        logging.info(f"InferenceModel.predict - Classified patches, number: {len(patch_predictions)}")

        # 6. Aggregate predictions
        prob_list = self._aggregate_patch_pred(patch_predictions)
        logging.info(f"InferenceModel.predict - Aggregated predictions, number: {len(prob_list)}")
        
        return prob_list

    def predict(
        self,
        slide_filename: str,
    ) -> List[Dict[str, Union[Tuple[int, int], str]]]:
        """
        Inference on a slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to run inference on.

        Returns
        -------
            slide_annotations: List[Dict]
                A list of annotations for the slide.
        """
        prob_list = self.predict_proba(slide_filename)
        slide_preds = []
        for prob in prob_list:
            slide_preds.append({
                "coord": prob["coord"],
                "pred": np.argmax(prob["pred"])
            })

        return slide_preds
    
    @staticmethod
    def plot_annotations(
        slide_filename: str,
        slide_preds: List[Dict[str, Union[Tuple[int, int], str]]],
        roi_patch_size: int,
        num_classes: int,
        save_plot_path: str,
        is_prob: bool = False
    ):
        """
        Plot annotations on slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to plot annotations on.
            slide_annotations: List[Dict[str, Union[Tuple[int, int], str]]]
                A list of annotations for the slide.
            num_classes: int
                The number of classes.
            save_plot_path: str
                The path to save the plot to.
            is_prob: bool
                Whether the annotations are probabilities or not.
        """
        slide = openslide.OpenSlide(slide_filename)
        
        scale_factor = np.array(slide.level_dimensions[-1]) / np.array(slide.level_dimensions[0])
        scaled_patch_size = roi_patch_size * scale_factor
        if not is_prob: # plot annotations
            fig, ax = plt.subplots(1, 2, figsize=(20, 20))
            ax[0].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
            ax[0].set_title("Ground Truth")

            ax[1].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
            ax[1].set_title("Predictions")
            pred_mask = np.zeros(slide.level_dimensions[-1], dtype=np.uint8).T
            for slide_pred in slide_preds:
                coord = np.array(slide_pred["coord"]) * scale_factor
                pred = slide_pred["pred"]
                x, y = coord
                pred_mask[int[y]:int(y+scaled_patch_size), int[x]:int(x+scaled_patch_size)] = pred
            ax[1].imshow(pred_mask, alpha=0.3, cmap="Reds", vmin=0, vmax=num_classes-1)

            cmap = plt.cm.get_cmap('Reds', 4)
            fig.legend(
                handles=[
                    plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(num_classes)
                ],
                labels=[
                    LABELS2TYPE_MAP[i] for i in range(num_classes)
                ],
                loc="upper right"
            )

        else: # plot n_classes heatmaps
            fig, ax = plt.subplots(num_classes + 1, 1, figsize=(20, 20))
            ax[0].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
            ax[0].set_title("Ground Truth")

            for i in range(1, num_classes + 1):
                ax[i].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
                pred_mask = np.zeros(slide.level_dimensions[-1], dtype=np.uint8).T
                for slide_pred in slide_preds:
                    coord = np.array(slide_pred["coord"]) * scale_factor
                    pred = slide_pred["pred"][i-1]
                    x, y = coord
                    pred_mask[int[y]:int(y+scaled_patch_size), int[x]:int(x+scaled_patch_size)] = pred
                ax[i].imshow(pred_mask, alpha=0.3, cmap="gray", vmin=0, vmax=1)
                ax[i].set_title(f"Class {LABELS2TYPE_MAP[i]}")
                    
                
        plt.savefig(save_plot_path)
        plt.close()

    def _extract_patches_from_slide(self, slide: openslide.OpenSlide) -> np.ndarray:
        """
        Extract patches from a slide using the lowest resolution slide thumbnail.
        
        Parameters
        ----------
            slide: openslide.OpenSlide
                The slide to extract patches from.

        Returns
        -------
            slide_patches: np.ndarray
                The patches extracted from the slide.
        """
        logging.debug(f"InferenceModel._extract_patches_from_slide started")
        # get lowest resolution slide thumbnail
        slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
        slide_thumbnail = np.array(slide_thumbnail)

        # add padding to slide thumbnail so it mods slide_patch_size
        # pad at the bottom and right to simplify coordinates calculations later
        slide_thumbnail = np.pad(
            slide_thumbnail,
            (
                (0, self.slide_patch_size - slide_thumbnail.shape[0] % self.slide_patch_size),
                (0, self.slide_patch_size - slide_thumbnail.shape[1] % self.slide_patch_size),
                (0, 0)
            ),
            constant_values=255
        )

        slide_patches = patchify(
            slide_thumbnail,
            (self.slide_patch_size, self.slide_patch_size, 3),
            step=self.slide_patch_size
        )

        logging.debug(f"InferenceModel._extract_patches_from_slide ended")

        return slide_patches
    
    def _flag_majority_white(self, slide_patches: np.ndarray) -> np.ndarray:
        """
        Flag patches extracted from the lowest resolution thumbnail image that are majority white.
        
        Parameters
        ----------
            slide_patches: np.ndarray
                The patches to flag.
        
        Returns
        -------
            majority_white: np.ndarray
                A boolean array indicating whether a patch is majority white.
        """
        logging.debug(f"InferenceModel._flag_majority_white started")
        flatten_patches = slide_patches.reshape(-1, self.slide_patch_size, 3)
        flatten_bw_patches = cv2.cvtColor(flatten_patches, cv2.COLOR_BGR2GRAY)
        flatten_bw_patches = flatten_bw_patches.reshape(slide_patches.shape[0]*slide_patches.shape[1], self.slide_patch_size, self.slide_patch_size)
        majority_white = flatten_bw_patches.mean(axis=0) > self.white_threshold
        majority_white = majority_white.reshape(slide_patches.shape[0], slide_patches.shape[1])

        # majority_white = np.zeros(slide_patches.shape[:2], dtype=np.bool)
        # for row in range(slide_patches.shape[0]):
        #     for col in range(slide_patches.shape[1]):
        #         patch = slide_patches[row, col, 0]
        #         bw_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        #         majority_white[row, col] = bw_patch.mean() > self.white_threshold

        logging.debug(f"InferenceModel._flag_majority_white ended")

        return majority_white
    
    def _extract_bboxes(
        self,
        slide_patches: np.ndarray,
        majority_white: np.ndarray
    ) -> List[Dict[str, Union[int, np.ndarray]]]:
        """
        Extract bounding boxes from patches.

        Runs object detection on patches that are not majority white.
        
        Parameters
        ----------
            slide_patches: np.ndarray
                The patches to extract bounding boxes from.
            majority_white: np.ndarray
                A boolean array indicating whether a patch is majority white.

        Returns
        -------
            bbox_predictions: List[Dict[str, Union[int, np.ndarray]]]
                A list of dictionaries containing the row and column of the patch and the predicted bounding boxes.
        """
        logging.debug(f"InferenceModel._extract_bboxes started")
        bbox_predictions = []
        self.object_detection_model.to(self.device)
        self.object_detection_model.eval()
        for row in slide_patches.shape[0]:
            for col in slide_patches.shape[1]:
                if majority_white[row][col]:
                    continue
                patch = slide_patches[row, col, 0]
                patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
                pred = self.object_detection_model(patch)
                pred_boxes = pred["boxes"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                keep = torchvision.ops.nms(
                    torch.from_numpy(pred_boxes),
                    torch.from_numpy(pred_scores),
                    self.box_nms_threshold
                )

                # filter out boxes with low scores
                keep = keep[pred_scores[keep] > self.box_score_threshold]
                # filter out boxes with small areas
                keep = keep[(pred_boxes[keep, 2] - pred_boxes[keep, 0]) * (pred_boxes[keep, 3] - pred_boxes[keep, 1]) > self.box_area_threshold]

                bbox_predictions.append({
                    "row": row,
                    "col": col,
                    "pred_boxes": pred_boxes[keep]
                })

        logging.debug(f"InferenceModel._extract_bboxes ended")

        return bbox_predictions
    
    def _scale_bboxes(
        self,
        slide: openslide.OpenSlide,
        bbox_predictions: List[Dict[str, Union[int, np.ndarray]]]
    ) -> List[np.ndarray]:
        """
        Scale bounding boxes to original image size.

        Multiply the coordinates of the bounding boxes by the scaling factor between the lowest resolution and highest resolution.
        
        Parameters
        ----------
            slide: openslide.OpenSlide
                The slide to scale the bounding boxes to.
            bbox_predictions: List[Dict[str, Union[int, np.ndarray]]]
                A list of dictionaries containing the row and column of the patch and the predicted bounding boxes.
                
        Returns
        -------
            scaled_bboxes: List[np.ndarray]
                A list of scaled bounding boxes.        
        """
        logging.debug(f"InferenceModel._scale_bboxes started")
        scaled_bboxes = []
        # scaling factor between lowest resolution and highest resolution in [width, height]
        scale_factor = np.array(slide.level_dimensions[0]) / np.array(slide.level_dimensions[-1])
        for bbox_prediction in bbox_predictions:
            boxes = bbox_prediction["pred_boxes"]
            # scale boxes to original image size using absolute coordinates
            offset = np.array([
                bbox_prediction["row"],
                bbox_prediction["col"]
            ]) * self.slide_patch_size
            boxes = boxes + offset
            boxes = boxes.reshape(-1, 2, 2) * scale_factor[::-1]
            boxes = boxes.reshape(-1, 4)
            scaled_bboxes.append(boxes)

        logging.debug(f"InferenceModel._scale_bboxes ended")

        return scaled_bboxes
    
    def _extract_patches_from_rois(
        self,
        slide: openslide.OpenSlide,
        scaled_bboxes: List[np.ndarray]
    ) -> List[Dict[str, Union[Tuple[int, int], np.ndarray]]]:
        """
        Extract patches from bounding boxes.

        Padding is added to the bounding boxes so that:
        1. The width and height of the bounding boxes are multiples of roi_patch_size.
        2. The min x and y coordinates are multiples of roi_patch_size.
        3. The new coordinates are within the slide dimensions.

        Parameters
        ----------
            slide: openslide.OpenSlide
                The slide to extract patches from.
            scaled_bboxes: List[np.ndarray]
                A list of scaled bounding boxes.

        Returns
        -------
            roi_patches: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of dictionaries containing the patches and the coordinates of the padded bounding boxes.
        """
        logging.debug(f"InferenceModel._extract_patches_from_rois started")
        roi_patches = []
        for bboxes in scaled_bboxes:
            for bbox in bboxes:
                # calculate coordinates of roi patch so it doesn't go out of bounds and mods patch_size
                min_coord, max_coord = pad_roi_coordinates(
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    slide.dimensions,
                    self.roi_patch_size,
                    self.roi_patch_size,
                    for_inference=True
                )

                roi_region = slide.read_region(
                    min_coord,
                    0,
                    tuple(np.subtract(max_coord, min_coord))
                )
                roi_region = np.array(roi_region)

                region_patches = patchify(
                    roi_region,
                    (self.roi_patch_size, self.slide_patch_size, 3),
                    step=self.roi_patch_size
                )
                roi_patches.append({
                    "patches": region_patches,
                    "min_coord": min_coord,
                    "max_coord": max_coord
                })

        logging.debug(f"InferenceModel._extract_patches_from_rois ended")

        return roi_patches

    def _classify_patches(
        self,
        roi_patches: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
    ) -> List[Dict[str, Union[Tuple[int, int], np.ndarray]]]:
        """
        Classify patches extracted from the predicted bounding boxes.

        Parameters
        ----------
            roi_patches: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of dictionaries containing the patches and the coordinates of the padded bounding boxes.

        Returns
        -------
            patch_predictions: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of dictionaries containing the probabilities of each class and the coordinates of the padded bounding boxes.
        """
        logging.debug(f"InferenceModel._classify_patches started")
        patch_predictions = []
        self.classifier_model.to(self.device)
        self.classifier_model.eval()
        for roi_patch in roi_patches:
            patches = roi_patch["patches"]
            patches = patches.reshape(-1, 3, self.roi_patch_size, self.roi_patch_size)
            patches = torch.from_numpy(patches).to(self.device)
            patches = patches.float() / 255.0
            pred = self.classifier_model(patches)
            pred = pred.softmax(dim=1)
            pred = pred.cpu().numpy().reshape(roi_patches.shape[0], roi_patches.shape[1], -1)
            patch_predictions.append({
                "pred": pred,
                "min_coord": roi_patch["min_coord"],
                "max_coord": roi_patch["max_coord"]
            })

        logging.debug(f"InferenceModel._classify_patches ended")

        return patch_predictions
    
    def _aggregate_patch_pred(
        self,
        patch_predictions: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
    ) -> List[Dict[str, Union[Tuple[int, int], np.ndarray]]]:
        """
        Aggregate patch predictions by averaging predictions for each retrieved patch bounded by the predicted bounding boxes.
        
        Create the slide annotations in every roi_patch_size x roi_patch_size pixels patch.

        Slide annotations should have min x and y coordinates of multples of roi_patch_size
        
        Parameters
        ----------
            patch_predictions: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of dictionaries containing the probabilities of each class and the coordinates of the padded bounding boxes.
                
        Returns
        -------
            slide_annotations: List[Dict[str, Union[Tuple[int, int], np.ndarray]]]
                A list of annotations for the slide.
        """
        logging.debug(f"InferenceModel._aggregate_patch_pred started")
        pixel_predictions = {}
        for patch_prediction in patch_predictions:
            pred = patch_prediction["pred"]
            min_coord = patch_prediction["min_coord"] # (x_min, y_min)

            # compile list of bbox coordinates for each class
            for y in range(pred.shape[0]):
                for x in range(pred.shape[1]):
                    pixel_prediction = pred[y, x]
                    pixel_coord = (min_coord[0] + x*self.roi_patch_size, min_coord[1] + y*self.roi_patch_size)
                    if pixel_coord not in pixel_predictions:
                        pixel_predictions[pixel_coord] = [pixel_prediction[y, x], 1]
                    else:
                        pixel_predictions[pixel_coord][0] += pixel_prediction[y, x]
                        pixel_predictions[pixel_coord][1] += 1
        
        aggregate_predictions = []
        # average predictions for each pixel
        for coords, (pred_prob, count) in pixel_predictions.items():
            aggregate_predictions.append({
                "coord": coords,
                "pred": pred_prob / count
            })

        logging.debug(f"InferenceModel._aggregate_patch_pred ended")

        return aggregate_predictions