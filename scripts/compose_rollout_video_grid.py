#!/usr/bin/env python3
"""Compose equally sized rollout videos into a synchronized comparison grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--labels", nargs="+")
    return parser.parse_args()


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 132, min(image.width - 12, 330), 174), fill=(8, 8, 8))
    draw.text((24, 145), label, fill=(255, 215, 64))
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if args.columns <= 0 or args.fps <= 0:
        raise SystemExit("columns and fps must be positive")
    inputs = [path.expanduser().resolve() for path in args.inputs]
    labels = args.labels or [path.stem for path in inputs]
    if len(labels) != len(inputs):
        raise ValueError(f"labels/inputs mismatch: {len(labels)}/{len(inputs)}")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input videos: {missing}")
    readers = [imageio.get_reader(str(path)) for path in inputs]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    rows = math.ceil(len(readers) / args.columns)
    frame_count = 0
    try:
        while True:
            frames = []
            for reader in readers:
                try:
                    frames.append(np.asarray(reader.get_next_data()))
                except (IndexError, StopIteration):
                    frames = []
                    break
            if not frames:
                break
            frames = [add_label(frame, label) for frame, label in zip(frames, labels)]
            shape = frames[0].shape
            if any(frame.shape != shape for frame in frames):
                raise ValueError(f"input frame sizes do not match: {[frame.shape for frame in frames]}")
            blank = np.zeros_like(frames[0])
            frames.extend([blank] * (rows * args.columns - len(frames)))
            grid_rows = [
                np.concatenate(frames[row * args.columns : (row + 1) * args.columns], axis=1)
                for row in range(rows)
            ]
            writer.append_data(np.concatenate(grid_rows, axis=0))
            frame_count += 1
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    print(f"grid: {output}")
    print(f"inputs: {len(inputs)}, layout: {rows}x{args.columns}, frames: {frame_count}")


if __name__ == "__main__":
    main()
