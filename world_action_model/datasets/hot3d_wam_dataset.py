from __future__ import annotations

import io
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from giga_datasets.datasets.base_dataset import BaseDataset
from giga_datasets.datasets.dataset import load_config, register_dataset


ACTION_DIM = 53
BASE_SLICE = slice(0, 3)
LEFT_WRIST_SLICE = slice(3, 12)
RIGHT_WRIST_SLICE = slice(12, 21)
GRIPPER_SLICE = slice(21, 23)
LEFT_FINGERS_SLICE = slice(23, 38)
RIGHT_FINGERS_SLICE = slice(38, 53)
HOT3D_SUPERVISED_DIM_MASK = np.asarray(
    [False] * 3 + [True] * 18 + [False] * 2 + [True] * 30,
    dtype=np.bool_,
)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    row = np.asarray(rot6d, dtype=np.float32).reshape(6)
    a1, a2 = row[:3], row[3:]
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    a2 = a2 - float(np.dot(b1, a2)) * b1
    b2 = a2 / max(float(np.linalg.norm(a2)), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    return matrix[:, :2].T.reshape(6).astype(np.float32)


def _relative_wrist_pose(current: np.ndarray, following: np.ndarray, hand: int) -> np.ndarray:
    trans_slice = slice(0, 3) if hand == 0 else slice(3, 6)
    rot_slice = slice(6, 12) if hand == 0 else slice(12, 18)
    trans_now = current[trans_slice]
    trans_next = following[trans_slice]
    rot_now = rot6d_to_matrix(current[rot_slice])
    rot_next = rot6d_to_matrix(following[rot_slice])
    relative_translation = rot_now.T @ (trans_next - trans_now)
    relative_rotation = rot_now.T @ rot_next
    return np.concatenate([relative_translation, matrix_to_rot6d(relative_rotation)]).astype(np.float32)


def hot3d_state(lowdim: np.ndarray, presence: int) -> tuple[np.ndarray, np.ndarray]:
    lowdim = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    if lowdim.size != 116:
        raise ValueError(f"Expected HOT3D lowdim shape (116,), got {lowdim.shape}")
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    if presence & 1:
        state[LEFT_WRIST_SLICE] = np.concatenate([lowdim[0:3], lowdim[6:12]])
        state[LEFT_FINGERS_SLICE] = lowdim[18:33]
        mask[LEFT_WRIST_SLICE] = True
        mask[LEFT_FINGERS_SLICE] = True
    if presence & 2:
        state[RIGHT_WRIST_SLICE] = np.concatenate([lowdim[3:6], lowdim[12:18]])
        state[RIGHT_FINGERS_SLICE] = lowdim[33:48]
        mask[RIGHT_WRIST_SLICE] = True
        mask[RIGHT_FINGERS_SLICE] = True
    return state, mask


def hot3d_action(
    current: np.ndarray,
    following: np.ndarray,
    current_presence: int,
    following_presence: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    following = np.asarray(following, dtype=np.float32).reshape(-1)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    mask = HOT3D_SUPERVISED_DIM_MASK.copy()
    left_valid = bool((current_presence & 1) and (following_presence & 1))
    right_valid = bool((current_presence & 2) and (following_presence & 2))
    if left_valid:
        action[LEFT_WRIST_SLICE] = _relative_wrist_pose(current, following, hand=0)
        action[LEFT_FINGERS_SLICE] = following[18:33]
    else:
        mask[LEFT_WRIST_SLICE] = False
        mask[LEFT_FINGERS_SLICE] = False
    if right_valid:
        action[RIGHT_WRIST_SLICE] = _relative_wrist_pose(current, following, hand=1)
        action[RIGHT_FINGERS_SLICE] = following[33:48]
    else:
        mask[RIGHT_WRIST_SLICE] = False
        mask[RIGHT_FINGERS_SLICE] = False
    return action, mask


@register_dataset
class HOT3DWAMDataset(BaseDataset):
    """Indexed HOT3D WebDataset windows for 53-D GigaWorld pretraining."""

    def __init__(
        self,
        index_path: str,
        data_size: int | None = None,
        action_horizon: int = 48,
        image_frame_offsets: tuple[int, ...] | list[int] = (0, 12, 24, 36, 48),
        robotype: str = "hot3d_human",
        prompt_embed_dim: int = 4096,
        max_prompt_len: int = 64,
    ) -> None:
        super().__init__(data_path=str(Path(index_path).resolve().parent))
        self.index_path = str(Path(index_path).resolve())
        self.data_size = None if data_size is None else int(data_size)
        self.action_horizon = int(action_horizon)
        self.image_frame_offsets = tuple(int(x) for x in image_frame_offsets)
        self.robotype = str(robotype)
        self.prompt_embed_dim = int(prompt_embed_dim)
        self.max_prompt_len = int(max_prompt_len)
        self._connection: sqlite3.Connection | None = None
        self._tar_fds: dict[str, int] = {}

    @classmethod
    def load(cls, data_or_config: str | dict) -> BaseDataset:
        config = load_config(data_or_config)
        return cls(**{k: v for k, v in config.items() if not k.startswith("_")})

    def open(self) -> None:
        if self._connection is None:
            if not os.path.isfile(self.index_path):
                raise FileNotFoundError(f"HOT3D WAM index not found: {self.index_path}")
            uri = f"file:{self.index_path}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for fd in self._tar_fds.values():
            os.close(fd)
        self._tar_fds.clear()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        if self.data_size is not None:
            return self.data_size
        self.open()
        row = self._connection.execute("SELECT COUNT(*) FROM windows").fetchone()
        self.data_size = int(row[0])
        return self.data_size

    def _read_member(self, shard_path: str, offset: int, size: int) -> bytes:
        fd = self._tar_fds.get(shard_path)
        if fd is None:
            fd = os.open(shard_path, os.O_RDONLY)
            self._tar_fds[shard_path] = fd
        payload = os.pread(fd, int(size), int(offset))
        if len(payload) != int(size):
            raise IOError(f"Short read from {shard_path}: expected {size}, got {len(payload)}")
        return payload

    def _get_data(self, index: int) -> dict:
        window = self._connection.execute(
            "SELECT episode_id, start_frame FROM windows WHERE id=?",
            (int(index) + 1,),
        ).fetchone()
        if window is None:
            raise IndexError(index)
        episode_id, start_frame = int(window[0]), int(window[1])
        end_frame = start_frame + self.action_horizon
        rows = self._connection.execute(
            """
            SELECT frame_index, shard_path, image_offset, image_size,
                   lowdim_offset, lowdim_size, presence
            FROM frames
            WHERE episode_id=? AND frame_index BETWEEN ? AND ?
            ORDER BY frame_index
            """,
            (episode_id, start_frame, end_frame),
        ).fetchall()
        if len(rows) != self.action_horizon + 1:
            raise RuntimeError(
                f"Broken HOT3D window episode={episode_id} start={start_frame}: "
                f"expected {self.action_horizon + 1} frames, got {len(rows)}"
            )

        lowdims, presences = [], []
        for row in rows:
            payload = self._read_member(row[1], row[4], row[5])
            lowdims.append(np.load(io.BytesIO(payload), allow_pickle=False).astype(np.float32, copy=False))
            presences.append(int(row[6]))

        images = []
        for offset in self.image_frame_offsets:
            row = rows[offset]
            payload = self._read_member(row[1], row[2], row[3])
            with Image.open(io.BytesIO(payload)) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))

        state, state_dim_mask = hot3d_state(lowdims[0], presences[0])
        action, action_dim_mask = zip(
            *[
                hot3d_action(lowdims[t], lowdims[t + 1], presences[t], presences[t + 1])
                for t in range(self.action_horizon)
            ]
        )
        return {
            "observation.images.ego_view": np.stack(images),
            "observation.state": state[None],
            "state_dim_mask": state_dim_mask[None],
            "action": np.stack(action),
            "action_dim_mask": np.stack(action_dim_mask),
            "action_loss_mask": np.asarray([m.any() for m in action_dim_mask], dtype=np.bool_),
            "t5_embedding": np.zeros((self.max_prompt_len, self.prompt_embed_dim), dtype=np.float32),
            "robotype": self.robotype,
            "data_path": self.data_path,
        }
