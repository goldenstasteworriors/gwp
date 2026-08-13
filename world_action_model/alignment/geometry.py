"""Geometry metrics for action-to-video consequence alignment.

The functions in this module deliberately operate on NumPy arrays. They are
evaluation utilities: training-time policy gradients remain in
``consequence.py`` and do not depend on OpenCV or a robot simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if x.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def binary_mask_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute occupancy agreement without assuming equal mask area."""

    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError(f"Mask shapes differ: {predicted.shape} vs {target.shape}")

    intersection = np.logical_and(predicted, target).sum(dtype=np.float64)
    union = np.logical_or(predicted, target).sum(dtype=np.float64)
    pred_area = predicted.sum(dtype=np.float64)
    target_area = target.sum(dtype=np.float64)
    pred_centroid = _mask_centroid(predicted)
    target_centroid = _mask_centroid(target)
    centroid_px = float(np.linalg.norm(pred_centroid - target_centroid))
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2.0 * intersection / (pred_area + target_area))
        if pred_area + target_area
        else 1.0,
        "centroid_px": centroid_px,
        "area_ratio": float(pred_area / target_area) if target_area else float("nan"),
    }


def copy_initial_mask_baseline(initial_mask: np.ndarray, num_horizons: int) -> np.ndarray:
    """Repeat the observed robot mask as an action-independent future baseline."""

    initial_mask = np.asarray(initial_mask)
    if initial_mask.ndim != 2:
        raise ValueError(f"Expected a 2D initial mask, got {initial_mask.shape}")
    if num_horizons <= 0:
        raise ValueError(f"num_horizons must be positive, got {num_horizons}")
    return np.repeat((initial_mask > 0)[None], int(num_horizons), axis=0)


def per_horizon_mean_mask_baseline(
    training_masks: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Build an action- and observation-independent mask prior from training data."""

    training_masks = np.asarray(training_masks)
    if training_masks.ndim != 4:
        raise ValueError(
            f"Expected training masks with shape [N, T, H, W], got {training_masks.shape}"
        )
    if training_masks.shape[0] == 0:
        raise ValueError("At least one training trajectory is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return (training_masks > 0).mean(axis=0) >= threshold


def _masked_overlap_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | bool]:
    predicted = np.logical_and(predicted, valid)
    target = np.logical_and(target, valid)
    true_positive = np.logical_and(predicted, target).sum(dtype=np.float64)
    false_positive = np.logical_and(predicted, np.logical_not(target)).sum(dtype=np.float64)
    false_negative = np.logical_and(np.logical_not(predicted), target).sum(dtype=np.float64)
    predicted_area = predicted.sum(dtype=np.float64)
    target_area = target.sum(dtype=np.float64)
    union = true_positive + false_positive + false_negative
    denominator = predicted_area + target_area
    return {
        # An empty union means both masks correctly predict no occupancy.  Area
        # counts remain in the result so aggregate reports can expose how often
        # this convention is used rather than hiding it in a mean score.
        "iou": float(true_positive / union) if union else 1.0,
        "dice": float(2.0 * true_positive / denominator) if denominator else 1.0,
        "precision": float(true_positive / predicted_area) if predicted_area else float("nan"),
        "recall": float(true_positive / target_area) if target_area else float("nan"),
        "predicted_pixels": int(predicted_area),
        "target_pixels": int(target_area),
        "valid_pixels": int(valid.sum(dtype=np.int64)),
        "empty_union": bool(union == 0),
    }


def counterfactual_mask_metrics(
    predicted_masks: np.ndarray,
    target_masks: np.ndarray,
    initial_mask: np.ndarray,
    *,
    valid_masks: np.ndarray | None = None,
    target_gripper_masks: np.ndarray | None = None,
    horizons: Sequence[int] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate whether future robot masks depend on a counterfactual action.

    Besides full-mask overlap, this reports motion relative to the observed
    initial robot mask and newly occupied pixels.  Those two views prevent the
    large static base/arm region from making an action-independent predictor
    look useful.  ``target_gripper_masks`` is optional because some renderers
    provide only a full robot segmentation.

    Floating-point predictions are interpreted as probabilities and thresholded
    at ``threshold``.  Callers holding logits must apply a sigmoid first.
    """

    predicted_masks = np.asarray(predicted_masks)
    target_masks = np.asarray(target_masks)
    if predicted_masks.ndim != 3 or target_masks.ndim != 3:
        raise ValueError(
            "Expected predicted and target masks with shape [T, H, W], got "
            f"{predicted_masks.shape} and {target_masks.shape}"
        )
    if predicted_masks.shape != target_masks.shape:
        raise ValueError(
            f"Predicted and target mask shapes differ: {predicted_masks.shape} vs "
            f"{target_masks.shape}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    num_horizons, height, width = target_masks.shape
    initial_mask = np.asarray(initial_mask)
    if initial_mask.shape != (height, width):
        raise ValueError(
            f"Expected initial mask shape {(height, width)}, got {initial_mask.shape}"
        )

    if valid_masks is None:
        valid_masks = np.ones_like(target_masks, dtype=bool)
    else:
        valid_masks = np.asarray(valid_masks, dtype=bool)
        if valid_masks.shape == (height, width):
            valid_masks = np.broadcast_to(valid_masks, target_masks.shape)
        if valid_masks.shape != target_masks.shape:
            raise ValueError(
                f"Expected valid masks shape {(height, width)} or {target_masks.shape}, "
                f"got {valid_masks.shape}"
            )
    if np.any(valid_masks.sum(axis=(-2, -1)) == 0):
        raise ValueError("Every horizon must contain at least one valid pixel")

    if target_gripper_masks is not None:
        target_gripper_masks = np.asarray(target_gripper_masks)
        if target_gripper_masks.shape != target_masks.shape:
            raise ValueError(
                f"Target gripper mask shape {target_gripper_masks.shape} does not match "
                f"target robot masks {target_masks.shape}"
            )
        target_gripper_masks = target_gripper_masks > 0

    if horizons is None:
        horizon_values = list(range(num_horizons))
    else:
        horizon_values = [int(value) for value in horizons]
        if len(horizon_values) != num_horizons:
            raise ValueError(
                f"Expected {num_horizons} horizon labels, got {len(horizon_values)}"
            )

    predicted = (
        predicted_masks
        if predicted_masks.dtype == np.bool_
        else predicted_masks >= threshold
    )
    target = target_masks > 0
    initial = initial_mask > 0

    per_horizon: list[dict[str, Any]] = []
    for index, horizon in enumerate(horizon_values):
        valid = valid_masks[index]
        predicted_dynamic = np.logical_xor(predicted[index], initial)
        target_dynamic = np.logical_xor(target[index], initial)
        predicted_new = np.logical_and(predicted[index], np.logical_not(initial))
        target_new = np.logical_and(target[index], np.logical_not(initial))
        row: dict[str, Any] = {
            "horizon": horizon,
            "full": _masked_overlap_metrics(predicted[index], target[index], valid),
            "dynamic": _masked_overlap_metrics(predicted_dynamic, target_dynamic, valid),
            "new_occupancy": _masked_overlap_metrics(predicted_new, target_new, valid),
        }
        if target_gripper_masks is not None:
            gripper = np.logical_and(target_gripper_masks[index], valid)
            gripper_pixels = gripper.sum(dtype=np.float64)
            covered = np.logical_and(predicted[index], gripper).sum(dtype=np.float64)
            row["gripper_region_recall"] = (
                float(covered / gripper_pixels) if gripper_pixels else float("nan")
            )
            row["gripper_pixels"] = int(gripper_pixels)
        per_horizon.append(row)

    mean: dict[str, float] = {}
    for region in ("full", "dynamic", "new_occupancy"):
        for metric in ("iou", "dice", "precision", "recall"):
            values = np.asarray([row[region][metric] for row in per_horizon], dtype=np.float64)
            mean[f"{region}_{metric}"] = (
                float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
            )
        mean[f"{region}_empty_union_fraction"] = float(
            np.mean([row[region]["empty_union"] for row in per_horizon])
        )
    if target_gripper_masks is not None:
        gripper_values = np.asarray(
            [row["gripper_region_recall"] for row in per_horizon], dtype=np.float64
        )
        mean["gripper_region_recall"] = (
            float(np.nanmean(gripper_values))
            if np.any(np.isfinite(gripper_values))
            else float("nan")
        )

    return {
        "horizons": horizon_values,
        "per_horizon": per_horizon,
        "mean": mean,
    }


def estimate_background_homography(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_exclude_mask: np.ndarray | None = None,
    target_exclude_mask: np.ndarray | None = None,
    *,
    max_features: int = 3000,
    ratio_test: float = 0.75,
    ransac_threshold_px: float = 2.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate source-to-target camera motion using only static background.

    Robot pixels are excluded from feature detection. This prevents a correct
    robot motion from being mistaken for camera motion when compensating mild
    viewpoint jitter.
    """

    import cv2

    source_rgb = _as_uint8_rgb(source_rgb)
    target_rgb = _as_uint8_rgb(target_rgb)
    if source_rgb.shape != target_rgb.shape:
        raise ValueError(f"Image shapes differ: {source_rgb.shape} vs {target_rgb.shape}")

    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((9, 9), dtype=np.uint8)

    def feature_mask(exclude: np.ndarray | None) -> np.ndarray | None:
        if exclude is None:
            return None
        exclude = np.asarray(exclude, dtype=np.uint8)
        exclude = cv2.dilate(exclude, kernel, iterations=2)
        return np.where(exclude > 0, 0, 255).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
    source_keypoints, source_desc = orb.detectAndCompute(
        source_gray, feature_mask(source_exclude_mask)
    )
    target_keypoints, target_desc = orb.detectAndCompute(
        target_gray, feature_mask(target_exclude_mask)
    )
    identity = np.eye(3, dtype=np.float64)
    if source_desc is None or target_desc is None:
        return identity, {"status": "insufficient_descriptors", "matches": 0, "inliers": 0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [first for first, second in pairs if first.distance < ratio_test * second.distance]
    if len(good) < 4:
        return identity, {"status": "insufficient_matches", "matches": len(good), "inliers": 0}

    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good])
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good])
    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
    )
    if homography is None or not np.isfinite(homography).all():
        return identity, {"status": "homography_failed", "matches": len(good), "inliers": 0}
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inliers < 4:
        return identity, {"status": "insufficient_inliers", "matches": len(good), "inliers": inliers}
    return homography.astype(np.float64), {
        "status": "ok",
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inliers / len(good)),
    }


def warp_binary_mask(mask: np.ndarray, homography: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Warp a binary source mask into target image coordinates."""

    import cv2

    height, width = output_shape
    warped = cv2.warpPerspective(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(homography, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def register_and_compare_masks(
    source_rgb: np.ndarray,
    source_robot_mask: np.ndarray,
    target_rgb: np.ndarray,
    target_robot_mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Compare robot masks after background-only camera-jitter registration."""

    homography, registration = estimate_background_homography(
        source_rgb,
        target_rgb,
        source_exclude_mask=source_robot_mask,
        target_exclude_mask=target_robot_mask,
    )
    registered_mask = warp_binary_mask(source_robot_mask, homography, target_robot_mask.shape)
    return binary_mask_metrics(registered_mask, target_robot_mask), homography, registration


def project_world_points(
    points_world: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    vertical_fov_degrees: float,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with MuJoCo's camera convention into image pixels.

    ``camera_rotation_world`` maps camera-local axes to world axes. MuJoCo
    cameras look along local -Z and use local +Y as image up.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in xyz, got {points.shape}")
    original_shape = points.shape[:-1]
    points = points.reshape(-1, 3)
    position = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(camera_rotation_world, dtype=np.float64).reshape(3, 3)
    camera_points = (points - position) @ rotation
    depth = -camera_points[:, 2]

    height, width = image_shape
    focal = 0.5 * height / np.tan(np.deg2rad(vertical_fov_degrees) * 0.5)
    safe_depth = np.where(depth > 1e-8, depth, 1.0)
    pixels = np.stack(
        [
            0.5 * width + focal * camera_points[:, 0] / safe_depth,
            0.5 * height - focal * camera_points[:, 1] / safe_depth,
        ],
        axis=-1,
    )
    valid = (
        (depth > 1e-8)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels.reshape(*original_shape, 2), valid.reshape(original_shape)
