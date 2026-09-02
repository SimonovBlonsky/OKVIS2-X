#!/usr/bin/env python3
"""Shared mocap reference corrections used by EGO2 analyses."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


FINAL_BA_FILE = "okvis2-slam-calib-final-ba_trajectory.csv"
FIXED_DIAGNOSTIC_LEVER_M = np.asarray([-0.1195, -0.0035, 0.1563])
LEVER_AFFECTED_SEQUENCES = frozenset(
    {
        "20260805-122310",
        "20260805-123231",
        "20260805-123752",
    }
)


@dataclass(frozen=True)
class LeverEvaluation:
    reference_positions: np.ndarray
    estimate_positions: np.ndarray
    errors_m: np.ndarray
    rmse_m: float


def session_fixed_lever(
    sequence: str, fixed_lever_m: np.ndarray
) -> np.ndarray:
    lever = np.asarray(fixed_lever_m, dtype=float)
    if lever.shape != (3,):
        raise ValueError("fixed_lever_m must contain three values")
    if not np.isfinite(lever).all():
        raise ValueError("fixed_lever_m must be finite")
    return lever.copy() if sequence in LEVER_AFFECTED_SEQUENCES else np.zeros(3)


def correct_reference_positions(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    lever_m: np.ndarray,
) -> np.ndarray:
    positions_array = np.asarray(positions, dtype=float)
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    lever = np.asarray(lever_m, dtype=float)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError("positions must be N x 3")
    if quaternions.shape != (len(positions_array), 4):
        raise ValueError("quaternions must be N x 4")
    if lever.shape != (3,):
        raise ValueError("lever_m must contain three values")
    if not (
        np.isfinite(positions_array).all()
        and np.isfinite(quaternions).all()
        and np.isfinite(lever).all()
    ):
        raise ValueError("positions, quaternions, and lever must be finite")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    return positions_array + rotations.apply(
        np.broadcast_to(lever, positions_array.shape)
    )


def _rigid_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return (rotation @ source.T).T + translation


def apply_effective_lever(
    reference_positions: np.ndarray,
    reference_quaternions_wxyz: np.ndarray,
    estimate_positions: np.ndarray,
    lever_m: np.ndarray,
) -> LeverEvaluation:
    estimate = np.asarray(estimate_positions, dtype=float)
    corrected_reference = correct_reference_positions(
        reference_positions, reference_quaternions_wxyz, lever_m
    )
    if estimate.shape != corrected_reference.shape:
        raise ValueError("reference and estimate positions must have matching shapes")
    if not np.isfinite(estimate).all():
        raise ValueError("estimate positions must be finite")
    aligned_estimate = _rigid_align(estimate, corrected_reference)
    errors = np.linalg.norm(aligned_estimate - corrected_reference, axis=1)
    return LeverEvaluation(
        corrected_reference,
        aligned_estimate,
        errors,
        float(np.sqrt(np.mean(errors**2))),
    )
