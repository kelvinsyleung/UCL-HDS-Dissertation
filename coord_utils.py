from typing import Tuple

import numpy as np

def get_relative_coordinates(coordinates: np.ndarray, min_coord: Tuple) -> np.ndarray:
    """
    Given a list of coordinates, return the relative coordinates

    returns: list of relative coordinates
    """
    return np.subtract(coordinates, min_coord)

def pad_roi_coordinates(
    min_coord: Tuple[int, int],
    max_coord: Tuple[int, int],
    max_boundaries: Tuple[int, int],
    patch_size: int,
    step_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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

    return new_min_coord, new_max_coord