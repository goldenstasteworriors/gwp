"""Geometry metrics for action-to-video consequence alignment.

The functions in this module deliberately operate on NumPy arrays. They are
evaluation utilities: training-time policy gradients remain in
``consequence.py`` and do not depend on OpenCV or a robot simulator.
"""

from __future__ import annotations

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
