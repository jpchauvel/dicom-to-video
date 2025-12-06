#!/usr/bin/env python
import argparse
import pathlib

import cv2
import matplotlib.pyplot as plt
from pydicom import dcmread


def get_maximum_dimensions(images: list[pathlib.Path]) -> tuple[int, int]:
    prev_width = prev_height = 0
    for image in images:
        frame = cv2.imread(image.as_posix())
        if frame is not None:
            height, width, _ = frame.shape
            if width > prev_width:
                prev_width = width
            if height > prev_height:
                prev_height = height
    return prev_width, prev_height


def main(args: argparse.Namespace) -> None:
    # The path to the example "ct" dataset included with pydicom
    input_path: pathlib.Path = pathlib.Path(args.input_path).expanduser()
    output_path: pathlib.Path = pathlib.Path(args.output_path).expanduser()
    output_file: pathlib.Path = pathlib.Path(args.output_file).expanduser()

    # Ensure output file has .avi extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(".avi")

    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        # Sort DICOM files by Instance Number to preserve order
        dicom_files = sorted(input_path.iterdir(), key=lambda f: f.name)
        dicom_data = []

        for f in dicom_files:
            ds = dcmread(f)
            dicom_data.append((f, ds))

        # Save images in sorted order
        for f, ds in dicom_data:
            # `arr` is a numpy.ndarray
            arr = ds.pixel_array
            plt.imsave(output_path / f"{f.name}.png", arr, cmap="gray")

        # Reconstruct the sorted image list in the same order
        images: list[pathlib.Path] = (
            [output_path / f"{f.name}.png" for f, _ in dicom_data]
        )

        # Get maximum dimensions across all images
        width, height = get_maximum_dimensions(images)

        # Create VideoWriter with proper parameters
        # Use MJPEG codec for .avi, 1 fps, color output
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        video = cv2.VideoWriter(
            output_file.as_posix(), fourcc, 1.0, (width, height)
        )

        for image in images:
            frame = cv2.imread(image.as_posix())
            if frame is not None:
                # Resize frame to match video dimensions
                # Use border padding to preserve aspect ratio
                h, w = frame.shape[:2]

                # Create a black canvas with the target dimensions
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
        "--output_path", required=True, help="Intermmediary directory path"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output AVI file path (without the extension)",
    )
    main(parser.parse_args())
