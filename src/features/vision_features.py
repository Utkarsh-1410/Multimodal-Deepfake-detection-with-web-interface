from __future__ import annotations

import numpy as np
import cv2
from typing import Dict


def compute_blink_metric(landmarks: np.ndarray) -> float:
    # Eye aspect ratio approximation using FaceMesh indices (simplified)
    # If landmarks not sufficient, return neutral
    if landmarks.shape[0] < 468:
        return 0.0
    # sample points: left eye (159,145,33,133), right eye (386,374,263,362)
    def ear(p1, p2, p3, p4) -> float:
        v = np.linalg.norm(landmarks[p1] - landmarks[p2]) + np.linalg.norm(landmarks[p3] - landmarks[p4])
        h = np.linalg.norm(landmarks[33] - landmarks[133]) + 1e-6
        return float(v / h)

    left = ear(159, 145, 160, 144)
    right = ear(386, 374, 387, 373)
    return float((left + right) * 0.5)


def compute_texture_stats(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    mean, std = cv2.meanStdDev(gray)
    return {"var_lap": var_lap, "mean": float(mean[0][0]), "std": float(std[0][0])}


def compute_region_attention_mask(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if landmarks.shape[0] < 468:
        mask[:] = 1.0
        return mask
    # emphasize eyes and mouth regions roughly
    eyes_idx = [33, 133, 362, 263]
    mouth_idx = [13, 14, 308, 78]
    pts = (landmarks * np.array([[w, h]])).astype(np.int32)
    for i in eyes_idx + mouth_idx:
        cx, cy = pts[i]
        cv2.circle(mask, (cx, cy), max(4, w // 30), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = mask / (mask.max() + 1e-6)
    return mask


