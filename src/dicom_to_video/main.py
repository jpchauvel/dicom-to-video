#!/usr/bin/env python
import argparse
import pathlib

import cv2
import numpy as np
from pydicom import dcmread


def get_maximum_dimensions(arrays: list[np.ndarray]) -> tuple[int, int]:
    """Get the maximum width and height across all image arrays."""
    max_width = max_height = 0
    for arr in arrays:
        height, width = arr.shape[:2]
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
    return max_width, max_height


def grayscale_to_bgr(gray_array: np.ndarray) -> np.ndarray:
    """Convert grayscale numpy array to BGR format for video writing."""
    # Normalize to 0-255 range if needed
    if gray_array.dtype != np.uint8:
        # Find min and max for proper scaling
        arr_min = float(gray_array.min())
        arr_max = float(gray_array.max())

        # Scale to 0-255 range
        if arr_max > arr_min:
            gray_normalized = (
                (gray_array.astype(np.float64) - arr_min)
                / (arr_max - arr_min)
                * 255
            ).astype(np.uint8)
        else:
            gray_normalized = np.zeros_like(gray_array, dtype=np.uint8)
        gray_array = gray_normalized

    # Convert grayscale to BGR (3 channels)
    return cv2.cvtColor(gray_array, cv2.COLOR_GRAY2BGR)


def main(args: argparse.Namespace) -> None:
    # The path to the example "ct" dataset included with pydicom
    input_path: pathlib.Path = pathlib.Path(args.input_path).expanduser()
    output_file: pathlib.Path = pathlib.Path(args.output_file).expanduser()

    # Ensure output file has .avi extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(".avi")

    if input_path.is_dir():
        # Sort DICOM files by filename to establish baseline order
        dicom_files = sorted(input_path.iterdir(), key=lambda f: f.name)
        dicom_data = []

        # Read all DICOM files and store with metadata
        for f in dicom_files:
            ds = dcmread(f)
            dicom_data.append((f, ds))

        # Extract pixel arrays in sorted order and convert to BGR
        image_arrays: list[np.ndarray] = []
        for f, ds in dicom_data:
            arr = ds.pixel_array
            bgr_array = grayscale_to_bgr(arr)
            image_arrays.append(bgr_array)

        # Get maximum dimensions across all images
        width, height = get_maximum_dimensions(image_arrays)

        # Create VideoWriter with proper parameters
        # Use MJPEG codec for .avi, 1 fps, color output
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        video = cv2.VideoWriter(
            output_file.as_posix(), fourcc, 1.0, (width, height)
        )

        # Write each frame to video
        for frame in image_arrays:
            h, w = frame.shape[:2]

            # Create a black canvas with padding to match video dimensions
            canvas = cv2.copyMakeBorder(
                frame,
                0,
                height - h,  # top, bottom
                0,
                width - w,  # left, right
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

            video.write(canvas)

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert DICOM medical sequential images to AVI"
    )
    parser.add_argument(
        "--input_path", required=True, help="DICOM directory path"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output AVI file path (without the extension)",
    )
    main(parser.parse_args())
