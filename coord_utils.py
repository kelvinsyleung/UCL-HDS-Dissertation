from typing import Tuple, List

import numpy as np

def get_relative_coordinates(coordinates: np.ndarray, min_coord: Tuple) -> np.ndarray:
    """
    Given a list of coordinates, return the relative coordinates

    Parameters
    ----------
        coordinates: np.ndarray
            list of coordinates
        min_coord: Tuple
            minimum coordinate of the roi (x_min, y_min)

    Returns
    -------
        list of relative coordinates
    """
    return np.subtract(coordinates, min_coord)

def get_absolute_coordinates(coordinates: np.ndarray, min_coord: Tuple) -> np.ndarray:
    """
    Given a list of coordinates, return the absolute coordinates

    Parameters
    ----------
        coordinates: np.ndarray
            list of coordinates
        min_coord: Tuple
            minimum coordinate of the roi (x_min, y_min)

    Returns
    -------
        list of absolute coordinates
    """
    return np.add(coordinates, min_coord)

def get_bbox_by_shape(coordinates: List, shape_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of coordinates and geometry type, return the bounding box of the roi

    Parameters
    ----------
        coordinates: List
            list of coordinates
        shape_type: str
            type of shape (Polygon or MultiPolygon)

    Returns
    -------
        min_coord: np.ndarray
            minimum coordinate of the roi (x_min, y_min)
        max_coord: np.ndarray
            maximum coordinate of the roi (x_max, y_max)
    """
    coordinates_arr = np.array([])
    if shape_type == "Polygon":
        # first coordinate is the outer polygon, the rest are holes
        coordinates_arr = np.array(coordinates[0]).squeeze()
    elif shape_type == "MultiPolygon":
        for polygon in coordinates:
            coordinates_arr = np.array(polygon[0]).squeeze()
            if coordinates_arr.shape[0] > 4:
                break
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    min_coord = np.floor(np.min(coordinates_arr, axis=0)).astype(int)
    max_coord = np.ceil(np.max(coordinates_arr, axis=0)).astype(int)
    return tuple(min_coord), tuple(max_coord)

def pad_roi_coordinates(
    min_coord: Tuple[int, int],
    max_coord: Tuple[int, int],
    max_boundaries: Tuple[int, int],
    patch_size: int,
    step_size: int,
    for_inference: bool = False
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given a roi, pad the roi to fit the patch size and step size
    If for_inference is True, the roi will be padded to a coordinate that is divisible by patch size

    Parameters
    ----------
        min_coord: Tuple[int, int]
            minimum coordinate of the roi (x_min, y_min)
        max_coord: Tuple[int, int]
            maximum coordinate of the roi (x_max, y_max)
        max_boundaries: Tuple[int, int]
            maximum boundaries of the image (x_max, y_max)
        patch_size: int
            patch size
        step_size: int
            step size
        for_inference: bool
            whether the roi is for inference

    Returns
    -------
        new_min_coord: Tuple[int, int]
            new minimum coordinate of the roi (x_min, y_min)
        new_max_coord: Tuple[int, int]
            new maximum coordinate of the roi (x_max, y_max)
    """
    new_min_coord = min_coord
    new_max_coord = max_coord
    # if the roi is smaller than patch size, pad the roi
    if (new_max_coord[0] - new_min_coord[0]) < patch_size or (new_max_coord[1] - new_min_coord[1]) < patch_size:
        width_pad = patch_size - (new_max_coord[0] - new_min_coord[0])
        height_pad = patch_size - (new_max_coord[1] - new_min_coord[1])
        new_min_coord = (
            max(new_min_coord[0] - width_pad//2, 0),
            max(new_min_coord[1] - height_pad//2, 0)
        )
        new_max_coord = (
            min(new_max_coord[0] + width_pad//2, max_boundaries[0]),
            min(new_max_coord[1] + height_pad//2, max_boundaries[1])
        )
    
    # check if the patch is still smaller than patch size
    # if yes, add padding to the other side of the patch
    if (new_max_coord[0] - new_min_coord[0]) < patch_size or (new_max_coord[1] - new_min_coord[1]) < patch_size:
        if new_max_coord[0] == max_boundaries[0]:
            width_pad = patch_size - (new_max_coord[0] - new_min_coord[0])
            new_min_coord = (
                max(new_min_coord[0] - width_pad, 0),
                new_min_coord[1]
            )
        else:
            width_pad = patch_size - (new_max_coord[0] - new_min_coord[0])
            new_max_coord = (
                min(new_max_coord[0] + width_pad, max_boundaries[0]),
                new_max_coord[1]
            )

        if new_max_coord[1] == max_boundaries[1]:
            height_pad = patch_size - (new_max_coord[1] - new_min_coord[1])
            new_min_coord = (
                new_min_coord[0],
                max(new_min_coord[1] - height_pad, 0)
            )
        else:
            height_pad = patch_size - (new_max_coord[1] - new_min_coord[1])
            new_max_coord = (
                new_max_coord[0],
                min(new_max_coord[1] + height_pad, max_boundaries[1])
            )

    # if the roi does not mod step size, add padding to the sides of the patch
    if (new_max_coord[0] - new_min_coord[0]) % step_size != 0:
        width_pad = step_size - ((new_max_coord[0] - new_min_coord[0]) % step_size)
        new_min_coord = (
            max(new_min_coord[0] - width_pad//2, 0),
            new_min_coord[1]
        )
        new_max_coord = (
            min(new_max_coord[0] + width_pad//2, max_boundaries[0]),
            new_max_coord[1]
        )

    if (new_max_coord[1] - new_min_coord[1]) % step_size != 0:
        height_pad = step_size - ((new_max_coord[1] - new_min_coord[1]) % step_size)
        new_min_coord = (
            new_min_coord[0],
            max(new_min_coord[1] - height_pad//2, 0)
        )
        new_max_coord = (
            new_max_coord[0],
            min(new_max_coord[1] + height_pad//2, max_boundaries[1])
        )

    # check if the patch still does not mod step size
    # if yes, add padding to the other side of the patch
    if (new_max_coord[0] - new_min_coord[0]) % step_size != 0:
        if new_max_coord[0] == max_boundaries[0]:
            width_pad = step_size - ((new_max_coord[0] - new_min_coord[0]) % step_size)
            new_min_coord = (
                max(new_min_coord[0] - width_pad, 0),
                new_min_coord[1]
            )
        else:
            width_pad = step_size - ((new_max_coord[0] - new_min_coord[0]) % step_size)
            new_max_coord = (
                min(new_max_coord[0] + width_pad, max_boundaries[0]),
                new_max_coord[1]
            )

    if (new_max_coord[1] - new_min_coord[1]) % step_size != 0:
        if new_max_coord[1] == max_boundaries[1]:
            height_pad = step_size - ((new_max_coord[1] - new_min_coord[1]) % step_size)
            new_min_coord = (
                new_min_coord[0],
                max(new_min_coord[1] - height_pad, 0)
            )
        else:
            height_pad = step_size - ((new_max_coord[1] - new_min_coord[1]) % step_size)
            new_max_coord = (
                new_max_coord[0],
                min(new_max_coord[1] + height_pad, max_boundaries[1])
            )


    if for_inference:
        # x-axis
        # try shifting the roi if the x-coordinate does not mod patch size
        if new_min_coord[0] % patch_size != 0:
            width_shift = new_min_coord[0] % patch_size
            if new_min_coord[0] - width_shift >= 0:
                # shift the roi to the left
                new_min_coord = (
                    new_min_coord[0] - width_shift,
                    new_min_coord[1]
                )
                new_max_coord = (
                    new_max_coord[0] - width_shift,
                    new_max_coord[1]
                )
            elif new_max_coord[0] + width_shift <= max_boundaries[0]:
                # shift the roi to the right
                new_min_coord = (
                    new_min_coord[0] + width_shift,
                    new_min_coord[1]
                )
                new_max_coord = (
                    new_max_coord[0] + width_shift,
                    new_max_coord[1]
                )
            else:
                raise ValueError("Cannot shift roi to fit patch size")

        #  check against original min and max x-coordinates
        if new_min_coord[0] > min_coord[0]:
            width_pad = patch_size - (new_min_coord[0] % patch_size)
            new_min_coord = (
                new_min_coord[0] - width_pad,
                new_min_coord[1]
            )
        if new_max_coord[0] < max_coord[0]:
            width_pad = patch_size - (new_max_coord[0] % patch_size)
            new_max_coord = (
                new_max_coord[0] + width_pad,
                new_max_coord[1]
            )

        # y-axis
        # try shifting the roi to the top if the y-coordinate does not mod patch size
        if new_min_coord[1] % patch_size != 0:
            height_shift = new_min_coord[1] % patch_size
            if new_min_coord[1] - height_shift >= 0:
                # shift the roi to the top
                new_min_coord = (
                    new_min_coord[0],
                    new_min_coord[1] - height_shift
                )
                new_max_coord = (
                    new_max_coord[0],
                    new_max_coord[1] - height_shift
                )
            elif new_max_coord[1] + height_shift <= max_boundaries[1]:
                # shift the roi to the bottom
                new_min_coord = (
                    new_min_coord[0],
                    new_min_coord[1] + height_shift
                )
                new_max_coord = (
                    new_max_coord[0],
                    new_max_coord[1] + height_shift
                )
            else:
                raise ValueError("Cannot shift roi to fit patch size")
            
        #  check against original min and max y-coordinates
        if new_min_coord[1] > min_coord[1]:
            height_pad = patch_size - (new_min_coord[1] % patch_size)
            new_min_coord = (
                new_min_coord[0],
                new_min_coord[1] - height_pad
            )
        if new_max_coord[1] < max_coord[1]:
            height_pad = patch_size - (new_max_coord[1] % patch_size)
            new_max_coord = (
                new_max_coord[0],
                new_max_coord[1] + height_pad
            )

    return new_min_coord, new_max_coord