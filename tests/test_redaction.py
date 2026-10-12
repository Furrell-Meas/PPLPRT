import unittest
import tempfile
import os

import cv2
import numpy as np

from redaction import (
    MediaFile,
    ValidationError,
    filter_detections,
    expand_bounding_box,
    apply_gaussian_blur
)


class TestRedaction(unittest.TestCase):

    # ============================================================
    # TEST 1: IMAGE VALIDATION
    # ============================================================

    def test_valid_input(self):
        """
        Expected:
        - A valid JPG image is accepted.
        - validate() returns the loaded image.
        - Returned image has the expected dimensions.
        """

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as f:
            path = f.name

        try:
            # Deterministic image: completely black 100x100 image
            image = np.zeros(
                (100, 100, 3),
                dtype=np.uint8
            )

            cv2.imwrite(path, image)

            media = MediaFile(path, "test.jpg")

            result = media.validate()

            # Expected output
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (100, 100, 3))

        finally:
            os.remove(path)


    def test_invalid_input(self):
        """
        Expected:
        - An unsupported file is rejected.
        - ValidationError is raised.
        """

        with tempfile.NamedTemporaryFile(
            suffix=".txt",
            delete=False
        ) as f:
            path = f.name
            f.write(b"This is not an image.")

        try:
            media = MediaFile(path, "test.txt")

            # Expected output:
            # Validation must raise ValidationError.
            with self.assertRaises(ValidationError):
                media.validate()

        finally:
            os.remove(path)


    # ============================================================
    # TEST 2: FILTER DETECTIONS
    # ============================================================

    def test_filter_detections(self):
        """
        Expected:
        - Detections below 0.50 are removed.
        - Detections equal to or above 0.50 are retained.
        - The exact expected detections are returned.
        """

        detections = [
            (10, 10, 50, 20, 0.90),
            (20, 20, 40, 20, 0.30),
            (30, 30, 60, 25, 0.75),
            (40, 40, 70, 30, 0.49),
            (50, 50, 80, 35, 0.50)
        ]

        result = filter_detections(
            detections,
            confidence_threshold=0.50
        )

        # Expected output:
        # 0.90, 0.75 and 0.50 are retained.
        expected = [
            (10, 10, 50, 20, 0.90),
            (30, 30, 60, 25, 0.75),
            (50, 50, 80, 35, 0.50)
        ]

        self.assertEqual(result, expected)


    # ============================================================
    # TEST 3: EXPAND BOUNDING BOX
    # ============================================================

    def test_expand_bounding_box(self):
        """
        Expected:
        - Bounding box is expanded by 15%.
        - Result matches the expected coordinates.
        """

        image_shape = (100, 100, 3)

        result = expand_bounding_box(
            40, 40, 20, 10,
            image_shape,
            padding_percent=0.15
        )

        # Expectation
        #
        # Original:
        # x = 40, y = 40
        # width = 20, height = 10
        #
        # Expected expanded box:
        # (37, 39, 63, 51)

        expected = (37, 39, 63, 51)

        self.assertEqual(result, expected)


    def test_expand_bounding_box_near_edge(self):
        """
        Expected:
        - Bounding box is expanded.
        - Expanded coordinates are clipped to the image boundary.
        - No coordinate falls outside the image.
        """

        image_shape = (100, 100, 3)

        # Bounding box close to the bottom-right edge
        result = expand_bounding_box(
            85, 85, 15, 15,
            image_shape,
            padding_percent=0.15
        )

        x1, y1, x2, y2 = result

        # Expected output:
        # All coordinates must remain within image boundaries.
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)

        self.assertLessEqual(x2, 100)
        self.assertLessEqual(y2, 100)

        # The box must remain valid.
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)


    # ============================================================
    # TEST 4: GAUSSIAN BLUR
    # ============================================================

    def test_gaussian_blur_inside_region(self):
        """
        Expected:
        - Pixels inside the selected region are changed by blurring.
        - The result inside the selected region is NOT identical
          to the original region.
        """

        # Create a deterministic image.
        # The selected region contains a sharp pattern rather than
        # random values.
        image = np.zeros(
            (100, 100, 3),
            dtype=np.uint8
        )

        # Create a deterministic black/white pattern inside
        # the region that will be blurred.
        image[40:60, 40:50] = 255
        image[40:60, 50:60] = 0

        original = image.copy()

        result = apply_gaussian_blur(
            image,
            40, 40, 60, 60
        )

        # Extract the region before and after blurring.
        original_region = original[40:60, 40:60]
        result_region = result[40:60, 40:60]

        # Expected:
        # The selected region must change after Gaussian blur.
        self.assertFalse(
            np.array_equal(
                original_region,
                result_region
            )
        )


    def test_gaussian_blur_outside_region(self):
        """
        Expected:
        - Pixels outside the selected region remain unchanged.
        """

        # Deterministic image.
        image = np.zeros(
            (100, 100, 3),
            dtype=np.uint8
        )

        # Selected region
        image[40:60, 40:60] = 255

        # Outside region has a different deterministic value.
        image[0:40, 0:100] = 100
        image[60:100, 0:100] = 150
        image[40:60, 0:40] = 200
        image[40:60, 60:100] = 50

        original = image.copy()

        result = apply_gaussian_blur(
            image,
            40, 40, 60, 60
        )

        # Check the four areas outside the blur region.

        # Above
        self.assertTrue(
            np.array_equal(
                original[:40, :],
                result[:40, :]
            )
        )

        # Below
        self.assertTrue(
            np.array_equal(
                original[60:, :],
                result[60:, :]
            )
        )

        # Left
        self.assertTrue(
            np.array_equal(
                original[40:60, :40],
                result[40:60, :40]
            )
        )

        # Right
        self.assertTrue(
            np.array_equal(
                original[40:60, 60:],
                result[40:60, 60:]
            )
        )


# ================================================================
# RUN UNIT TESTS
# ================================================================

if __name__ == "__main__":
    unittest.main()