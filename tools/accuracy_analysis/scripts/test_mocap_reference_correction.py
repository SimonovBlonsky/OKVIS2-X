#!/usr/bin/env python3

import unittest

import numpy as np

from tools.accuracy_analysis.scripts import mocap_reference_correction as correction


class MocapReferenceCorrectionTest(unittest.TestCase):

    def test_session_fixed_lever_only_affects_confirmed_sequences(self):
        fixed = np.asarray([-0.1195, -0.0035, 0.1563])

        for sequence in (
            "20260805-122310",
            "20260805-123231",
            "20260805-123752",
        ):
            np.testing.assert_array_equal(
                correction.session_fixed_lever(sequence, fixed), fixed
            )

        np.testing.assert_array_equal(
            correction.session_fixed_lever("20260805-114334", fixed),
            np.zeros(3),
        )

    def test_correct_reference_positions_applies_body_fixed_lever(self):
        positions = np.zeros((2, 3))
        quaternions_wxyz = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        corrected = correction.correct_reference_positions(
            positions,
            quaternions_wxyz,
            np.asarray([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(
            corrected,
            np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            atol=1e-12,
        )

    def test_apply_effective_lever_rigidly_aligns_estimate(self):
        reference = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        quaternions_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
        lever = np.asarray([0.1, -0.2, 0.3])
        corrected_reference = reference + lever
        estimate = corrected_reference + np.asarray([2.0, -3.0, 4.0])

        result = correction.apply_effective_lever(
            reference,
            quaternions_wxyz,
            estimate,
            lever,
        )

        np.testing.assert_allclose(result.reference_positions, corrected_reference)
        np.testing.assert_allclose(result.estimate_positions, corrected_reference)
        self.assertLess(result.rmse_m, 1e-12)

    def test_rejects_invalid_lever_shape(self):
        with self.assertRaisesRegex(ValueError, "lever"):
            correction.session_fixed_lever(
                "20260805-122310", np.asarray([0.1, 0.2])
            )


if __name__ == "__main__":
    unittest.main()
