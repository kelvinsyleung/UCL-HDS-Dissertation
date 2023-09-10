import os
from typing import List, Dict, Union, Tuple
import logging
from pathlib import Path
from shutil import rmtree

import torchstain
import numpy as np
import matplotlib.pyplot as plt

import geojson
from patchify import patchify
from shapely.geometry import shape, Polygon, box
from shapely.ops import unary_union
import networkx as nx

import cv2
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from coord_utils import pad_roi_coordinates
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
        slide_white_threshold: float = 240, roi_white_threshold: float = 230,
        box_nms_threshold: float = 0.3,
        top_k_boxes: int = 10,
        class_prob_threshold: float = 0.75,
        num_classes: int = 3,
        classifier_batch_size: int = 128,
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
            classifier_mag: str
                The magnification of the patches to use for the classifier model.
            object_detection_cspace: str
                The colour space to use for the object detection model.
            norm_path: str
                The path to the normalisation files.
            slide_patch_size: int
                The size of the patches to extract from the slide.
            roi_patch_size: int
                The size of the patches to extract from the bounding boxes.
            slide_white_threshold: float
                The threshold to flag a slide level patch as majority white.
            roi_white_threshold: float
                The threshold to flag a roi level patch as majority white.
            box_nms_threshold: float
                The threshold to use for non-maximum suppression.
            top_k_boxes: int
                The number of top bounding boxes to use.
            class_prob_threshold: float
                The threshold of the multilabel probabilities to determine if the slide contains a class.

                Higher severity class precedes lower, even if the proportion of the lower severity class is greater.
            num_classes: int
                The number of classes. Currently only supports 3 classes.
        """
        self.classifier_model = classifier_model
        self.object_detection_model = object_detection_model
        self.device = device
        self.slide_patch_size = slide_patch_size
        self.roi_patch_size = roi_patch_size
        self.classifier_cspace = classifier_cspace
        self.obj_detect_cspace = obj_detect_cspace
        self.resize_roi_patch = True if classifier_mag == "20x" else False
        self.slide_white_threshold = slide_white_threshold
        self.roi_white_threshold = roi_white_threshold
        self.box_nms_threshold = box_nms_threshold
        self.top_k_boxes = top_k_boxes
        self.class_prob_threshold = class_prob_threshold
        self.num_classes = num_classes
        self.classifier_batch_size = classifier_batch_size

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

    def predict_logits(
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
                A dictionary containing the patches' min coordinates and the corresponding logits of each class.

        Steps
        -----
        1. Get bounding boxes
        2. Extracts patches from bounding boxes
        3. Runs classifier on patches
        4. Remove temporary files
        """
        Path("./inference").mkdir(parents=True, exist_ok=True)
        # 1. Get bounding boxes
        slide, scaled_bboxes = self.get_roi_bboxes(slide_filename)

        # 2. Extracts patches from bounding boxes
        self._extract_patches_from_rois(slide, scaled_bboxes)
        logging.info(
            f"InferenceModel.predict - Extracted roi patches by scaled coordinates and saved to disk")

        # 3. Runs classifier on patches
        logits_dict = self._classify_patches()
        logging.info(f"InferenceModel.predict - Classified patches")

        # 4. Remove temporary files
        rmtree("./inference")

        return logits_dict

    def get_roi_bboxes(self, slide_filename: str) -> Tuple[openslide.OpenSlide, List[np.ndarray]]:
        """
        Inference on a slide to get bounding boxes.

        Parameters
        ----------
            slide_filename: str
                The filename of the slide to run inference on.

        Returns
        -------
            slide: openslide.OpenSlide
                The slide object.
            scaled_bboxes: List[np.ndarray]
                A list of the bounding boxes' coordinates.

        Steps
        -----
        1. Extract patches from slide
        2. Flag slides that are majority white
        3. Runs object detection to get bounding boxes
        4. scale bounding boxes to absolute coordinates of the original image
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
        logging.debug(f"InferenceModel.predict - slide patches indices:")
        logging.debug(
            f"InferenceModel.predict - {[(bbox_prediction['row'], bbox_prediction['col']) for bbox_prediction in bbox_predictions]}"
        )

        # 4. scale bounding boxes to absolute coordinates of the original image
        scaled_bboxes = self._scale_bboxes_coord(slide, bbox_predictions)
        logging.info(
            f"InferenceModel.predict - Scaled bounding boxes coordinates")
        logging.debug(
            f"InferenceModel.predict - Scaled bounding boxes coordinates:")
        logging.debug(f"InferenceModel.predict - {scaled_bboxes}")
        return slide, scaled_bboxes

    def predict(
        self,
        slide_filename: str
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
                key: "roi_probs", value: A dictionary containing the patches' min coordinates and the probabilities of each class.
        """
        # get patch level logits
        logits_dict = self.predict_logits(slide_filename)

        if len(logits_dict) == 0:
            return {
                "roi_logits": {},
                "roi_probs": {},
                "roi_preds": {},
                "slide_class": 0
            }

        roi_probs = {}
        for coord in logits_dict.keys():
            roi_probs[coord] = torch.softmax(
                torch.from_numpy(logits_dict[coord]), dim=0).numpy()

        # get patch level predictions by taking the argmax of the logits
        roi_preds = {}
        for coord in logits_dict.keys():
            roi_preds[coord] = np.argmax(logits_dict[coord])

        # compute whole slide level prediction ver1
        # patch_class_counts = np.bincount(list(roi_preds.values()))
        # patch_class_counts = np.pad(
        #     patch_class_counts, (0, self.num_classes - patch_class_counts.shape[0]), mode="constant")
        # slide_class = patch_class_counts.argmax()

        # compute whole slide level prediction ver2
        # take the mean of the logits of all the patches
        mean_logits = np.array(list(logits_dict.values())).mean(axis=0)

        # convert logits to probabilities as a multilabel problem
        slide_multilabels = torch.sigmoid(torch.from_numpy(mean_logits))

        # threshold the probabilities, flip the tensor to get satisfied class from the highest severity to lowest
        slide_class = 2 - \
            torch.flip(slide_multilabels > self.class_prob_threshold,
                       dims=(0,)).int().argmax().item()

        slide_pred = {
            "roi_logits": logits_dict,
            "roi_probs": roi_probs,
            "roi_preds": roi_preds,
            "slide_class": slide_class
        }

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
        num_classes: int = 3
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

        scale_factor = slide.level_downsamples[0] / slide.level_downsamples[-1]
        scaled_patch_size = roi_patch_size * scale_factor

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        ax[0].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
        ax[0].set_title("Predictions")
        pred_mask = np.zeros(slide.level_dimensions[-1], dtype=np.uint8).T
        for coord, label in roi_preds.items():
            x, y = coord
            scaled_x, scaled_y = x * scale_factor, y * scale_factor
            pred_mask[
                int(scaled_y):int(scaled_y+scaled_patch_size),
                int(scaled_x):int(scaled_x+scaled_patch_size)
            ] = label + 1
        ax[0].imshow(
            pred_mask, alpha=0.65, cmap="Blues",
            vmin=0, vmax=num_classes + 1
        )

        cmap = plt.cm.get_cmap('Blues', 4)
        fig.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(1, num_classes + 1)
            ],
            labels=[
                LABELS2TYPE_MAP[i] for i in range(0, num_classes)
            ],
            loc="upper right"
        )

        ax[1].imshow(slide.get_thumbnail(slide.level_dimensions[-1]))
        ax[1].set_title("Ground Truth")

        cmap = plt.cm.get_cmap('Blues', 4)
        for feature in gt_annotations.features:
            label = NAME2TYPELABELS_MAP[feature.properties["classification"]["name"]]
            coord = np.array(feature.geometry.coordinates[0]) * scale_factor
            poly = Polygon(coord)
            x, y = poly.exterior.xy
            ax[-1].plot(x, y, color=cmap(label+1), linewidth=2)

        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()

    @staticmethod
    def calculate_roi_metrics(
        scaled_bboxes: List[np.ndarray],
        slide_total_area: float,
        gt_annotations: geojson.FeatureCollection
    ) -> Dict[str, float]:
        """
        Calculate ROI metrics for a slide.

        Parameters
        ----------
            scaled_bboxes: List[np.ndarray]
                A list of the bounding boxes' coordinates.
            gt_annotations: geojson.FeatureCollection
                A list of ground truth annotations for the slide.

        Returns
        -------
            metrics: Dict[str, float]
                A dictionary containing the metrics.
        """
        # convert scaled bboxes to shapely polygons
        scaled_bboxes = np.concatenate(scaled_bboxes, axis=0)
        scaled_bboxes = [box(*bbox) for bbox in scaled_bboxes]

        # convert ground truth annotations to shapely polygons
        gt_annotations = [shape(feature.geometry) for feature in gt_annotations.features]

        # calculate intersection area
        intersection_areas = []
        for scaled_bbox in scaled_bboxes:
            intersection_areas.append(
                [scaled_bbox.intersection(gt_annotation).area for gt_annotation in gt_annotations]
            )

        # calculate ground truth total area
        pred_total_area = sum([scaled_bbox.area for scaled_bbox in scaled_bboxes])
        gt_total_area = sum([gt_annotation.area for gt_annotation in gt_annotations])


        # using intersection and union to calculate specificity and sensitivity
        intersection_areas = np.array(intersection_areas)

        tp = intersection_areas.sum()
        p = gt_total_area

        tn = slide_total_area - pred_total_area - p + tp
        n = slide_total_area - gt_total_area

        specificity = tn / n
        sensitivity = tp / p

        metrics = {
            "specificity": specificity,
            "sensitivity": sensitivity
        }

        return metrics

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
        flatten_bw_patches = cv2.cvtColor(flatten_patches, cv2.COLOR_RGB2GRAY)
        flatten_bw_patches = flatten_bw_patches.reshape(
            slide_patches.shape[0]*slide_patches.shape[1],
            self.slide_patch_size*self.slide_patch_size
        )
        majority_white = flatten_bw_patches.mean(
            axis=1) > self.slide_white_threshold
        majority_white = majority_white.reshape(
            slide_patches.shape[0],
            slide_patches.shape[1]
        )

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
        with torch.no_grad():
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

                    if self.obj_detect_cspace == "CIELAB":
                        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)

                    patch = self.roi_transform(image=patch)["image"]

                    patch = patch.float().unsqueeze(0).to(self.device) / 255.0
                    pred = self.object_detection_model(patch)
                    pred_boxes = pred[0]["boxes"].cpu().detach().numpy()
                    pred_scores = pred[0]["scores"].cpu().detach().numpy()

                    nms_filtered_indices = torchvision.ops.nms(
                        torch.from_numpy(pred_boxes),
                        torch.from_numpy(pred_scores),
                        self.box_nms_threshold
                    )
                    keep = nms_filtered_indices[:self.top_k_boxes]

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

    def _scale_bboxes_coord(
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
        scale_factor = int(
            slide.level_downsamples[-1] / slide.level_downsamples[0])

        for bbox_prediction in bbox_predictions:
            boxes = bbox_prediction["pred_boxes"]
            row = bbox_prediction["row"]
            col = bbox_prediction["col"]
            # scale boxes to original image size using absolute coordinates
            offset = np.array([col, row, col, row]) * self.slide_patch_size
            boxes = boxes + offset
            boxes = boxes.reshape(-1, 2, 2) * scale_factor
            boxes = boxes.reshape(-1, 4)
            scaled_bboxes.append(boxes)

        logging.debug(f"InferenceModel._scale_bboxes ended")

        return scaled_bboxes

    def _extract_patches_from_rois(
        self,
        slide: openslide.OpenSlide,
        scaled_bboxes: List[np.ndarray]
    ):
        """
        Extract patches from bounding boxes. Then resize and save to disk to reduce memory usage.

        Parameters
        ----------
            slide: openslide.OpenSlide
                The slide to extract patches from.
            scaled_bboxes: List[np.ndarray]
                A list of scaled bounding boxes.
        """
        logging.debug(f"InferenceModel._extract_patches_from_rois started")
        roi_patches_saved = set()
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
                level = mag_factor - 1
                size = (
                    np.subtract(max_coord, min_coord) / (
                        slide.level_downsamples[level] /
                        slide.level_downsamples[0]
                    ) + 1  # add 1 to include the last pixel so it is a multiple of 256
                ).astype(int)
                roi_region = slide.read_region(min_coord, level, size)
                roi_region = np.array(roi_region)

                region_patches = patchify(
                    roi_region,
                    (
                        int(self.roi_patch_size / mag_factor),
                        int(self.roi_patch_size / mag_factor),
                        3
                    ),
                    step=int(self.roi_patch_size / mag_factor)
                )

                for row in range(region_patches.shape[0]):
                    for col in range(region_patches.shape[1]):
                        min_patch_coord = (
                            min_coord[0] + col * self.roi_patch_size,
                            min_coord[1] + row * self.roi_patch_size
                        )
                        if min_patch_coord in roi_patches_saved:
                            continue

                        patch = region_patches[row, col, 0]

                        # if patch is majority white, skip
                        if cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).mean() > self.roi_white_threshold:
                            continue

                        patch = cv2.resize(patch, (256, 256))
                        np.save(
                            f"./inference/{min_patch_coord[0]}_{min_patch_coord[1]}.npy",
                            patch
                        )

                        roi_patches_saved.add(min_patch_coord)

        logging.debug(f"InferenceModel._extract_patches_from_rois ended")

    def _classify_patches(
        self
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
        with torch.no_grad():
            self.classifier_model.to(self.device)
            self.classifier_model.eval()
            roi_patch_paths = list(Path("./inference").glob("*.npy"))

            if len(roi_patch_paths) == 0:
                return patch_predictions

            # create batches of patches
            batches = np.array_split(
                roi_patch_paths,
                np.ceil(len(roi_patch_paths) / self.classifier_batch_size)
            )

            roi_patch_predictions = []
            for batch in tqdm(batches):
                tensor_patches = torch.zeros(len(batch), 3, 256, 256)
                for i, path in enumerate(batch):
                    patch = np.load(path)
                    try:
                        patch, _, _ = self.stain_normaliser.normalize(patch)
                    except Exception:
                        pass

                    if self.classifier_cspace == "CIELAB":
                        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)

                    tensor_patches[i] = self.patch_transform(image=patch)[
                        "image"]
                tensor_patches = tensor_patches.to(self.device)
                pred = self.classifier_model(tensor_patches)
                pred = pred.cpu().detach().numpy()

                roi_patch_predictions.extend(pred)

                for i, path in enumerate(batch):
                    patch_predictions[tuple(
                        map(int, path.stem.split("_")))] = pred[i]

        logging.debug(f"InferenceModel._classify_patches ended")
        self.classifier_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return patch_predictions
