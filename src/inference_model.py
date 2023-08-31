import os
from typing import List, Dict, Union, Tuple
import logging

import torchstain
import numpy as np
import matplotlib.pyplot as plt

import geojson
from patchify import patchify
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
import networkx as nx

import cv2
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from coord_utils import pad_roi_coordinates
from log_utils import setup_logging
from class_mapping import LABELS2TYPE_MAP, NAME2TYPELABELS_MAP

OPENSLIDE_PATH = r'C:/openslide/openslide-win64/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class InferenceModel:
    def __init__(
        self, classifier_model: nn.Module, object_detection_model: nn.Module, device: str,
        classifier_cspace: str = "RGB", classifier_mag: str = "20x",
        obj_detect_cspace: str = "RGB",
        norm_path: str = "./data/norms",
        slide_patch_size: int = 512, roi_patch_size: int = 512,
        white_threshold: float = 230,
        box_nms_threshold: float = 0.15,
        box_score_threshold: float = 0.05,
        class_prop_threshold: float = 0.01,
        num_classes: int = 4
    ):
        """
        Parameters
        ----------
            classifier_model: nn.Module
                The classifier model to use.
            object_detection_model: nn.Module
                The object detection model to use.
            device: str
                The device to run inference on.
            classifier_cspace: str
                The colour space to use for the classifier model.
            object_detection_cspace: str
                The colour space to use for the object detection model.
            slide_patch_size: int
                The size of the patches to extract from the slide.
            roi_patch_size: int
                The size of the patches to extract from the bounding boxes.
            white_threshold: float
                The threshold to use to flag patches that are majority white.
            box_nms_threshold: float
                The threshold to use for non-maximum suppression.
            box_score_threshold: float
                The threshold to use to filter out boxes with low scores.
            box_area_threshold: float
                The threshold to use to filter out boxes with small areas.
            class_prop_threshold: float
                The threshold of the proportion of predicted rois of a class to flag the whole slide as that class.

                Higher severity class precedes lower, even if the proportion of the lower severity class is greater.
            num_classes: int
                The number of classes. Currently only supports 4 classes. 0 represents background.
        """
        self.classifier_model = classifier_model
        self.object_detection_model = object_detection_model
        self.device = device
        self.slide_patch_size = slide_patch_size
        self.roi_patch_size = roi_patch_size
        self.classifier_cspace = classifier_cspace
        self.obj_detect_cspace = obj_detect_cspace
        self.resize_roi_patch = True if classifier_mag == "20x" else False
        self.white_threshold = white_threshold
        self.box_nms_threshold = box_nms_threshold
        self.box_score_threshold = box_score_threshold
        self.class_prop_threshold = class_prop_threshold
        self.num_classes = num_classes

        rgb_mean = np.load(f"{norm_path}/rgb_mean.npy")
        rgb_std = np.load(f"{norm_path}/rgb_std.npy")
        logging.info("main - RGB mean and std loaded")

        cielab_mean = np.load(f"{norm_path}/cielab_mean.npy")
        cielab_std = np.load(f"{norm_path}/cielab_std.npy")
        logging.info("main - CIELAB mean and std loaded")

        self.roi_transform = A.Compose([
            A.Resize(512, 512),
            ToTensorV2()
        ])
        self.patch_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(rgb_mean if classifier_cspace == "RGB" else cielab_mean), std=(
                rgb_std if classifier_cspace == "RGB" else cielab_std)),
            ToTensorV2()
        ])

        # stain normalisation
        norm_img_arr = np.load(f"{norm_path}/stain_norm_img.npy")
        self.stain_normaliser = torchstain.normalizers.MacenkoNormalizer(
            backend='numpy')
        self.stain_normaliser.fit(norm_img_arr)

        logging.info("main - stain normalisation setup complete")

        setup_logging()

    def predict_proba(
        self,
        slide_filename: str
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Inference on a slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to run inference on.

        Returns
        -------
            prob_list: Dict[Tuple[int, int], np.ndarray]
                A dictionary containing the patches' min coordinates and the corresponding probabilities of each class.

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
        logging.info(
            f"InferenceModel.predict - Extracted slide patches, shape: {slide_patches.shape}"
        )

        # 2. Flag slides that are majority white
        majority_white = self._flag_majority_white(slide_patches)
        logging.info(
            f"InferenceModel.predict - Flagged majority white patches, proportion: {majority_white.mean()}"
        )

        # 3. Runs object detection to get bounding boxes
        bbox_predictions = self._extract_bboxes(slide_patches, majority_white)
        logging.info(
            f"InferenceModel.predict - Extracted bounding boxes from {len(bbox_predictions)} patches of lowest resolution"
        )
        logging.info(f"InferenceModel.predict - slide patches indices:")
        logging.info(
            f"InferenceModel.predict - {[(bbox_prediction['row'], bbox_prediction['col']) for bbox_prediction in bbox_predictions]}"
        )

        # 4. Extracts patches from bounding boxes
        # scale bounding boxes to absolute coordinates of the original image
        scaled_bboxes = self._scale_bboxes(slide, bbox_predictions)
        logging.info(f"InferenceModel.predict - Scaled bounding boxes:")
        logging.info(f"InferenceModel.predict - {scaled_bboxes}")

        # extract patches from bounding boxes
        roi_patches = self._extract_patches_from_rois(slide, scaled_bboxes)
        logging.info(
            f"InferenceModel.predict - Extracted roi patches coordinates")
        logging.info(f"InferenceModel.predict - {list(roi_patches.keys())}")

        # 5. Runs classifier on patches
        prob_dict = self._classify_patches(roi_patches)
        logging.info(f"InferenceModel.predict - Classified patches")

        return prob_dict

    def predict(
        self,
        slide_filename: str,
        return_prob: bool = False
    ) -> Dict[str, Union[Dict[Tuple[int, int], int], int]]:
        """
        Inference on a slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to run inference on.

        Returns
        -------
            slide_pred: Dict[str, Union[Dict[Tuple[int, int], int], int]]
                A dictionary containing the predictions for the slide.
                key: "roi_preds", value: A dictionary containing the patches' min coordinates  and the predicted class.
                key: "slide_class", value: The predicted class for the whole slide.
        """
        # get patch level probabilities
        prob_dict = self.predict_proba(slide_filename)

        if len(prob_dict) == 0:
            return {
                "roi_preds": [],
                "slide_class": 0
            }

        # get patch level predictions by taking the argmax of the probabilities
        roi_preds = {}
        for coord in prob_dict.keys():
            roi_preds[coord] = np.argmax(prob_dict[coord])

        # compute whole slide level prediction
        patch_class_counts = np.bincount(list(roi_preds.values()))
        patch_class_counts = np.pad(
            patch_class_counts, (0, self.num_classes - patch_class_counts.shape[0]), mode="constant")

        slide_class = 0
        # iterate from highest severity class to lowest
        for i in range(patch_class_counts.shape[0])[::-1]:
            # if the proportion of predicted rois of a class is greater than the threshold,
            # flag the whole slide as that class
            if patch_class_counts[i] / len(roi_preds) > self.class_prop_threshold:
                slide_class = i
                break

        slide_pred = {
            "roi_preds": roi_preds,
            "slide_class": slide_class
        }

        if return_prob:
            slide_pred["roi_probs"] = prob_dict

        return slide_pred

    @staticmethod
    def plot_annotations(
        slide_filename: str,
        roi_preds: Union[
            Dict[Tuple[int, int], int],
            Dict[Tuple[int, int], np.ndarray]
        ],
        gt_annotations: geojson.FeatureCollection,
        roi_patch_size: int,
        save_plot_path: str,
        is_prob: bool = False,
        num_classes: int = 4
    ):
        """
        Plot annotations on slide.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to plot annotations on.
            roi_preds: Union[
                Dict[Tuple[int, int], int],
                Dict[Tuple[int, int], np.ndarray]
            ]
                A list of annotations predictions for the slide.
            gt_annotations: geojson.FeatureCollection
                A list of ground truth annotations for the slide.
            roi_patch_size: int
                The size of the patches extracted from the bounding boxes.
            save_plot_path: str
                The path to save the plot to.
            is_prob: bool
                Whether the predictions are probabilities.
            num_classes: int
                The number of classes.
        """
        slide = openslide.OpenSlide(slide_filename)

        scale_factor = np.array(
            slide.level_dimensions[-1]) / np.array(slide.level_dimensions[0])
        scaled_patch_size = roi_patch_size * scale_factor

        if not is_prob:  # plot annotations
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))
            ax[0].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
            ax[0].set_title("Predictions")
            pred_mask = np.zeros(slide.level_dimensions[-1], dtype=np.uint8).T
            for coord, label in roi_preds.items():
                x, y = coord
                scaled_x, scaled_y = x * scale_factor[0], y * scale_factor[1]
                pred_mask[
                    int(scaled_y):int(scaled_y+scaled_patch_size[1]),
                    int(scaled_x):int(scaled_x+scaled_patch_size[0])
                ] = label
            ax[0].imshow(
                pred_mask, alpha=0.5, cmap="Blues",
                vmin=0, vmax=num_classes-1
            )

            cmap = plt.cm.get_cmap('Blues', 4)
            fig.legend(
                handles=[
                    plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(1, num_classes)
                ],
                labels=[
                    LABELS2TYPE_MAP[i] for i in range(1, num_classes)
                ],
                loc="upper right"
            )
        else:  # plot n_classes heatmaps
            fig, ax = plt.subplots(num_classes, 1, figsize=(12, 35))
            for i in range(1, num_classes):
                ax[i-1].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
                pred_mask = np.zeros(
                    slide.level_dimensions[-1], dtype=np.uint8).T
                for coord, label in roi_preds.items():
                    x, y = coord
                    pred_mask[
                        int(y):int(y+scaled_patch_size[1]),
                        int(x):int(x+scaled_patch_size[0])
                    ] = label
                ax[i-1].imshow(
                    pred_mask, alpha=0.5,
                    cmap="gray", vmin=0, vmax=1
                )
                ax[i-1].set_title(f"Class {LABELS2TYPE_MAP[i]} Probabilities")

        ax[-1].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
        ax[-1].set_title("Ground Truth")

        cmap = plt.cm.get_cmap('Blues', 4)
        for feature in gt_annotations.features:
            label = NAME2TYPELABELS_MAP[feature.properties["classification"]["name"]]
            scale_factor = np.array(
                slide.level_dimensions[-1]) / np.array(slide.level_dimensions[0])
            coord = np.array(feature.geometry.coordinates[0]) * scale_factor
            poly = Polygon(coord)
            x, y = poly.exterior.xy
            ax[-1].plot(x, y, color=cmap(label), linewidth=2)

        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()

    @staticmethod
    def calculate_metrics(
        roi_preds: Dict[Tuple[int, int], int],
        gt_annotations: geojson.FeatureCollection
    ) -> Dict[str, float]:
        """
        Calculate metrics for a slide.

        Parameters
        ----------
            roi_preds: Dict[Tuple[int, int], int]
                A list of annotations predictions for the slide.
            gt_annotations: geojson.FeatureCollection
                A list of ground truth annotations for the slide.

        Returns
        -------
            metrics: Dict[str, float]
                A dictionary containing the metrics.
        """
        # group bboxes by class
        bboxes_by_class = {}
        for coord, label in roi_preds.items():
            min_x, min_y = coord
            bbox = Polygon([
                (min_x, min_y),
                (min_x + 512, min_y),
                (min_x + 512, min_y + 512),
                (min_x, min_y + 512)
            ])
            bboxes_by_class.setdefault(label, []).append(bbox)

        # group adjacent bboxes within the same class
        grouped_bboxes_by_class = {}
        for label, bboxes in bboxes_by_class.items():
            # create graph
            G = nx.Graph()
            for i, bbox in enumerate(bboxes):
                G.add_node(i, bbox=bbox)

            # add edges between adjacent bboxes
            for i, bbox in enumerate(bboxes):
                for j, other_bbox in enumerate(bboxes):
                    if i == j:
                        continue
                    if bbox.intersects(other_bbox.buffer(0.5)):
                        G.add_edge(i, j)

            # group adjacent bboxes
            grouped_bboxes = []
            for component in nx.connected_components(G):
                grouped_bboxes.append(
                    unary_union([bboxes[i] for i in component])
                )

            grouped_bboxes_by_class[label] = grouped_bboxes

        dice_scores_per_class = {1: [], 2: [], 3: []}
        iou_scores_per_class = {1: [], 2: [], 3: []}

        for label, grouped_bboxes in grouped_bboxes_by_class.items():
            for pred_poly in grouped_bboxes:
                total_intersection_area = 0
                total_ground_truth_area = 0

                if label != 0:
                    for feature in gt_annotations.features:
                        if NAME2TYPELABELS_MAP[feature.properties["classification"]["name"]] == label:
                            gt_shape = shape(feature.geometry)
                            intersection_area = pred_poly.intersection(
                                gt_shape).area

                            if intersection_area > 0:
                                print(
                                    feature.properties["classification"]["name"], label)
                                print(feature.geometry["coordinates"])
                            total_intersection_area += intersection_area
                            total_ground_truth_area += gt_shape.area

                    union_area = pred_poly.area + total_ground_truth_area - total_intersection_area
                    total_area = pred_poly.area + total_ground_truth_area

                    dice_score = 2 * total_intersection_area / total_area
                    iou_score = total_intersection_area / union_area

                    dice_scores_per_class[label].append(dice_score)
                    iou_scores_per_class[label].append(iou_score)

        for label in dice_scores_per_class:
            if len(dice_scores_per_class[label]) == 0:
                dice_scores_per_class[label] = 0.
            else:
                dice_scores_per_class[label] = np.mean(
                    dice_scores_per_class[label])
            if len(iou_scores_per_class[label]) == 0:
                iou_scores_per_class[label] = 0.
            else:
                iou_scores_per_class[label] = np.mean(
                    iou_scores_per_class[label])

        return {
            "dice_scores": dice_scores_per_class,
            "iou_scores": iou_scores_per_class,
            "macro_dice_score": np.mean(list(dice_scores_per_class.values())),
            "macro_iou_score": np.mean(list(iou_scores_per_class.values()))
        }

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
                (0, self.slide_patch_size -
                 slide_thumbnail.shape[0] % self.slide_patch_size),
                (0, self.slide_patch_size -
                 slide_thumbnail.shape[1] % self.slide_patch_size),
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
        flatten_bw_patches = flatten_bw_patches.reshape(
            slide_patches.shape[0]*slide_patches.shape[1],
            self.slide_patch_size*self.slide_patch_size
        )
        majority_white = flatten_bw_patches.mean(axis=1) > self.white_threshold
        majority_white = majority_white.reshape(
            slide_patches.shape[0],
            slide_patches.shape[1]
        )

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
        for row in range(slide_patches.shape[0]):
            for col in range(slide_patches.shape[1]):
                if majority_white[row][col]:
                    continue
                patch = slide_patches[row, col, 0]
                try:
                    patch, _, _ = self.stain_normaliser.normalize(patch)
                except Exception as e:
                    logging.debug(
                        f"skipped normalising slide patch at {row}-{col}: {e}")

                if self.obj_detect_cspace == "RGB":
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                elif self.obj_detect_cspace == "CIELAB":
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

                patch = self.roi_transform(
                    image=slide_patches[row, col, 0])["image"]
                patch = patch.float().unsqueeze(0).to(self.device) / 255.0
                pred = self.object_detection_model(patch)
                pred_boxes = pred[0]["boxes"].cpu().detach().numpy()
                pred_scores = pred[0]["scores"].cpu().detach().numpy()

                nms_filtered_indices = torchvision.ops.nms(
                    torch.from_numpy(pred_boxes),
                    torch.from_numpy(pred_scores),
                    self.box_nms_threshold
                )

                nms_filtered_mask = np.zeros_like(pred_scores, dtype=bool)
                nms_filtered_mask[nms_filtered_indices] = 1

                # filter out boxes with low scores
                score_filtered_mask = pred_scores > self.box_score_threshold

                keep = nms_filtered_mask & score_filtered_mask

                bbox_predictions.append({
                    "row": row,
                    "col": col,
                    "pred_boxes": pred_boxes[keep]
                })

        logging.debug(f"InferenceModel._extract_bboxes ended")
        self.object_detection_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        scale_factor = np.array(
            slide.level_dimensions[0]
        ) / np.array(slide.level_dimensions[-1])

        for bbox_prediction in bbox_predictions:
            boxes = bbox_prediction["pred_boxes"]
            row = bbox_prediction["row"]
            col = bbox_prediction["col"]
            # scale boxes to original image size using absolute coordinates
            offset = np.array([col, row, col, row]) * self.slide_patch_size
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
    ) -> Dict[Tuple[int, int], np.ndarray]:
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
            roi_patches: Dict[Tuple[int, int], np.ndarray]
                A dictionary containing the patches' min coordinates and the corresponding array.
        """
        logging.debug(f"InferenceModel._extract_patches_from_rois started")
        roi_patches = {}
        mag_factor = 2 if self.resize_roi_patch else 1
        for bboxes in scaled_bboxes:
            for bbox in bboxes:
                # calculate coordinates of roi patch so it doesn't go out of bounds and mods patch_size
                min_coord, max_coord = pad_roi_coordinates(
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    slide.dimensions,
                    self.roi_patch_size * mag_factor,
                    self.roi_patch_size * mag_factor,
                    for_inference=True
                )

                roi_region = slide.read_region(
                    min_coord,
                    0,
                    tuple(np.subtract(max_coord, min_coord))
                ).convert("RGB")
                roi_region = np.array(roi_region)

                # resize according to classifier model's trained magnification
                if self.resize_roi_patch:
                    roi_region = cv2.resize(
                        roi_region, np.ceil([
                            roi_region.shape[1]/2,
                            roi_region.shape[0]/2
                        ]).astype(int)
                    )

                # print(roi_region.shape)
                region_patches = patchify(
                    roi_region,
                    (self.roi_patch_size, self.roi_patch_size, 3),
                    step=self.roi_patch_size
                )

                for row in range(region_patches.shape[0]):
                    for col in range(region_patches.shape[1]):
                        patch = region_patches[row, col, 0]

                        # if patch is majority white, skip
                        if cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean() > self.white_threshold:
                            continue
                        min_patch_coord = (
                            min_coord[0] + col*self.roi_patch_size,
                            min_coord[1] + row*self.roi_patch_size
                        )

                        if min_patch_coord not in roi_patches:
                            roi_patches[min_patch_coord] = patch

        logging.debug(f"InferenceModel._extract_patches_from_rois ended")

        return dict(sorted(roi_patches.items()))

    def _classify_patches(
        self,
        roi_patches: Dict[Tuple[int, int], np.ndarray]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Classify patches extracted from the predicted bounding boxes.

        Parameters
        ----------
            roi_patches: Dict[Tuple[int, int], np.ndarray]
                A dictionary containing the patches' min coordinates and the corresponding array.

        Returns
        -------
            patch_predictions: Dict[Tuple[int, int], np.ndarray]
                A dictionary containing the patches' min coordinates and the corresponding probabilities of each class.
        """
        logging.debug(f"InferenceModel._classify_patches started")
        patch_predictions = {}
        self.classifier_model.to(self.device)
        self.classifier_model.eval()
        roi_patch_keys = list(roi_patches.keys())
        roi_patch_values = list(roi_patches.values())
        roi_patch_values = np.array(roi_patch_values)
        roi_patch_values = roi_patch_values.reshape(
            -1, self.roi_patch_size, self.roi_patch_size, 3
        )

        if len(roi_patch_values) == 0:
            return patch_predictions

        # create batches of 16 patches
        batches = np.array_split(
            roi_patch_values,
            np.ceil(len(roi_patch_values) / 8)
        )

        roi_patch_predictions = []
        for batch in batches:
            tensor_patches = torch.zeros(batch.shape[0], 3, 256, 256)
            for i in range(batch.shape[0]):
                patch = batch[i]
                try:
                    patch, _, _ = self.stain_normaliser.normalize(patch)
                except Exception:
                    pass

                if self.classifier_cspace == "RGB":
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                elif self.classifier_cspace == "CIELAB":
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

                tensor_patches[i] = self.patch_transform(
                    image=batch[i])["image"]
            tensor_patches = tensor_patches.to(self.device)
            pred = self.classifier_model(tensor_patches)
            pred = pred.softmax(dim=1)
            pred = pred.cpu().detach().numpy().tolist()

            roi_patch_predictions.extend(pred)

        for key, pred in zip(roi_patch_keys, roi_patch_predictions):
            if key not in patch_predictions:
                patch_predictions[key] = pred

        logging.debug(f"InferenceModel._classify_patches ended")
        self.classifier_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return patch_predictions
