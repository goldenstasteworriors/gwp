#!/usr/bin/env python3
"""Run teacher-forced SONIC G1 evaluation through the policy ZMQ server.

At every replanning frame the client sends the current test-set image and
state, receives a complete 48-step action chunk, and records the chunk.  A
``--replan-steps 1`` rollout therefore represents closed-loop-style offline
evaluation: only the first predicted action is executed before the policy
re-observes the next test-set frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import pandas as pd
import zmq
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation

from convert_sonic_wrist_delta import load_robot_model, rot6d, wrist_poses


ACTION_HORIZON = 48
ACTION_DIM = 30
SERVER_IMAGE_SIZE = (320, 384)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default="grab the red bottle")
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def pack_default(obj: Any) -> dict:
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(type(obj).__name__)


def unpack_object(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def unique_episode_file(root: Path, episode: int, suffix: str) -> Path:
    matches = sorted(root.glob(f"**/episode_{episode:06d}.{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one episode_{episode:06d}.{suffix} under {root}, found {len(matches)}"
        )
    return matches[0]


def load_episode(
    dataset: Path,
    episode: int,
    sonic_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Path, Path]:
    parquet = unique_episode_file(dataset / "data", episode, "parquet")
    video = unique_episode_file(dataset / "videos", episode, "mp4")
    frame = pd.read_parquet(parquet)
    order_key = "frame_index" if "frame_index" in frame else "timestamp"
    frame = frame.sort_values(order_key).reset_index(drop=True)
    observed_eef = np.stack(frame["observation.eef_state"]).astype(np.float64)
    observed_state = np.stack(frame["observation.state"]).astype(np.float64)
    wbc_action = np.stack(frame["action.wbc"]).astype(np.float64)
    if observed_eef.shape != (len(frame), 14):
        raise ValueError(f"observation.eef_state must be [T,14], got {observed_eef.shape}")
    if observed_state.shape != (len(frame), 41) or wbc_action.shape != (len(frame), 41):
        raise ValueError(
            f"observation.state/action.wbc must be [T,41], got "
            f"{observed_state.shape}/{wbc_action.shape}"
        )

    state = np.empty((len(frame), ACTION_DIM), dtype=np.float32)
    for position_slice, quaternion_slice, output_start in (
        (slice(0, 3), slice(3, 7), 0),
        (slice(7, 10), slice(10, 14), 9),
    ):
        state[:, output_start : output_start + 3] = observed_eef[:, position_slice]
        quaternion = observed_eef[:, quaternion_slice]
        matrices = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
        state[:, output_start + 3 : output_start + 9] = np.stack(
            [rot6d(matrix) for matrix in matrices]
        )
    state[:, 18:24] = observed_state[:, 29:35]
    state[:, 24:30] = observed_state[:, 35:41]

    robot_model = load_robot_model(sonic_root)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    source_names = list(info["features"]["action.wbc"]["names"])
    target_position, target_rotation = wrist_poses(robot_model, wbc_action, source_names)
    action = np.zeros((len(frame), ACTION_DIM), dtype=np.float32)
    identity = rot6d(np.eye(3, dtype=np.float32))
    action[:, 3:9] = identity
    action[:, 12:18] = identity
    for index in range(len(frame) - 1):
        for side, output_start in ((0, 0), (1, 9)):
            current_rotation = target_rotation[index, side]
            relative_rotation = current_rotation.T @ target_rotation[index + 1, side]
            relative_position = current_rotation.T @ (
                target_position[index + 1, side] - target_position[index, side]
            )
            action[index, output_start : output_start + 3] = relative_position
            action[index, output_start + 3 : output_start + 9] = rot6d(relative_rotation)
        action[index, 18:24] = wbc_action[index + 1, 29:35] - wbc_action[index, 29:35]
        action[index, 24:30] = wbc_action[index + 1, 35:41] - wbc_action[index, 35:41]
    return frame, state, action, parquet, video


def save_rollout(
    output: Path,
    starts: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    state: np.ndarray,
    latency: np.ndarray,
    server_latency: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        starts=starts,
        pred=predictions,
        gt=ground_truth,
        state=state,
        latency=latency,
        server_latency=server_latency,
    )


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode < 0 or min(args.port, args.replan_steps, args.timeout_ms, args.save_every) <= 0:
        raise SystemExit("episode must be non-negative; numeric runtime arguments must be positive")
    if args.replan_steps > ACTION_HORIZON:
        raise SystemExit(f"replan-steps cannot exceed {ACTION_HORIZON}")

    dataset = args.dataset.expanduser().resolve()
    frame, state, action, parquet, video = load_episode(
        dataset, args.episode, args.sonic_root.expanduser().resolve()
    )
    starts = np.arange(0, len(frame) - ACTION_HORIZON, args.replan_steps, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"episode has only {len(frame)} frames; need more than {ACTION_HORIZON}")
    ground_truth = np.stack(
        [action[start : start + ACTION_HORIZON] for start in starts]
    ).astype(np.float32)
    predictions = np.empty_like(ground_truth)
    latency = np.empty(len(starts), dtype=np.float64)
    server_latency = np.empty(len(starts), dtype=np.float64)
    output = args.output.expanduser().resolve()
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != len(frame):
        raise ValueError(f"video/parquet frame mismatch: {video_frames} != {len(frame)}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{args.server}:{args.port}")
    decoded_frame = -1
    rgb = None
    wall_started = time.time()
    try:
        for request_index, start in enumerate(starts):
            while decoded_frame < start:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"video decode failed before frame {start}")
                decoded_frame += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = np.asarray(
                ImageOps.fit(
                    Image.fromarray(rgb), SERVER_IMAGE_SIZE, method=Image.Resampling.BILINEAR
                )
            )
            request = {
                "endpoint": "get_action",
                "data": {
                    "examples": [
                        {"image": [image], "lang": args.instruction, "state": state[start][None]}
                    ]
                },
            }
            begin = time.perf_counter()
            socket.send(msgpack.packb(request, default=pack_default))
            response = msgpack.unpackb(socket.recv(), object_hook=unpack_object)
            latency[request_index] = time.perf_counter() - begin
            if not response.get("ok"):
                raise RuntimeError(f"inference failed at frame {start}: {response.get('error')}")
            prediction = np.asarray(
                response["data"]["unnormalized_actions"], dtype=np.float32
            )
            if prediction.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(
                prediction
            ).all():
                raise ValueError(f"invalid prediction at frame {start}: {prediction.shape}")
            predictions[request_index] = prediction
            server_latency[request_index] = float(
                response["data"].get("inference_latency_s", latency[request_index])
            )
            completed = request_index + 1
            if completed % args.save_every == 0 or completed == len(starts):
                save_rollout(
                    partial,
                    starts[:completed],
                    predictions[:completed],
                    ground_truth[:completed],
                    state,
                    latency[:completed],
                    server_latency[:completed],
                )
                elapsed = time.time() - wall_started
                eta = elapsed / completed * (len(starts) - completed)
                print(
                    f"[{completed}/{len(starts)}] frame={start} "
                    f"server={server_latency[request_index]:.3f}s "
                    f"round_trip={latency[request_index]:.3f}s eta={eta:.1f}s",
                    flush=True,
                )
    finally:
        capture.release()
        socket.close()
        context.term()

    partial.replace(output)
    first_step_error = predictions[:, 0] - ground_truth[:, 0]
    summary = {
        "dataset": str(dataset),
        "parquet": str(parquet),
        "video": str(video),
        "episode": args.episode,
        "checkpoint": args.checkpoint,
        "instruction": args.instruction,
        "replan_steps": args.replan_steps,
        "frames": len(frame),
        "requests": len(starts),
        "prediction_shape": list(predictions.shape),
        "first_step_mse": float(np.mean(first_step_error**2)),
        "latency_seconds": {
            "server_mean": float(server_latency.mean()),
            "round_trip_mean": float(latency.mean()),
            "server_max": float(server_latency.max()),
        },
        "npz": output.name,
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
