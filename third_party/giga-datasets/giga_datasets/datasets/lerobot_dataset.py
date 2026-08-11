import json
import contextvars
import errno
import hashlib
import logging
import os
import re
import secrets
import shutil
import stat
import threading
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av
import datasets
import datasets.config as hf_datasets_config
import lerobot.datasets.lerobot_dataset as _lerobot_dataset_module
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torchvision
from filelock import FileLock, Timeout
from datasets.info import DatasetInfo
from datasets.naming import filenames_for_dataset_split
from datasets.packaged_modules.parquet.parquet import Parquet, ParquetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _OriginalLeRobotDatasetMetadata
from lerobot.datasets.utils import (
    embed_images,
    validate_episode_buffer,
    validate_frame,
)
from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames, get_safe_default_codec
from PIL import Image
from typing_extensions import override

from world_action_model.verified_io import (
    blocked_sigint,
    open_verified_regular_fd,
    read_verified_bytes,
    stat_identity,
    strict_json_loads,
)

from .base_dataset import BaseDataset, _get_data_worker_context
from .dataset import register_dataset


TemporalOffsetSpec = int | list[int] | dict[str, int | list[int]]
VideoKeySpec = str | Sequence[str]

_VERIFIED_METADATA_CONTEXT = contextvars.ContextVar(
    "gwp_verified_lerobot_metadata", default=None
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_NONCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}$")
_VERIFIED_STAT_KEYS = {
    "device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns"
}


@dataclass
class _VerifiedParquetConfig(ParquetConfig):
    verified_identity: str | None = None

    def create_config_id(self, config_kwargs: dict, custom_features=None) -> str:
        identity = config_kwargs.get("verified_identity")
        if not isinstance(identity, str) or not _SHA256_RE.fullmatch(identity):
            raise ValueError("verified parquet builder identity must be SHA-256")
        # The actual data_files remain dynamic /proc/self/fd paths so the
        # builder reads pinned source inodes.  They intentionally do not enter
        # this stable, content/config-derived builder key.
        return f"verified-{identity}"


class _VerifiedParquetBuilder(Parquet):
    BUILDER_CONFIG_CLASS = _VerifiedParquetConfig


def _is_file_descriptor_exhaustion(error: BaseException) -> bool:
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in {errno.EMFILE, errno.ENFILE}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _close_fd_once(fd: int) -> bool:
    """Attempt exactly one close; callers must relinquish the integer first."""

    try:
        os.close(fd)
    except OSError:
        # A failed close is never retried: on some kernels/errors the original
        # descriptor has already been released and its integer may be reused.
        return False
    return True


def _close_fd_sequence_once(fds: list[int]) -> None:
    """Consume and close every owned FD, then re-raise the first cancellation.

    The caller blocks normal SIGINT across consumption. A close that raises
    after releasing the integer is never retried.
    """

    first_error = None
    while fds:
        try:
            os.close(fds.pop())
        except OSError:
            # close(2) failure has ambiguous ownership; never retry the number.
            pass
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _directory_binding(file_stat: os.stat_result) -> dict[str, int]:
    return {
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "mode": int(file_stat.st_mode),
        "uid": int(file_stat.st_uid),
    }


class _CacheOwnership:
    """Single published owner for private-cache directories and Arrow FDs."""

    def __init__(self, *, parent_path: Path) -> None:
        self.parent_path = parent_path
        self._directory: dict[str, object] | None = None
        self.entries: dict[str, dict] = {}
        self.phase: tuple[str, int | None, datasets.Dataset | None] = (
            "building",
            None,
            None,
        )
        self.released = False
        self._arrow_fd_owner = _FDLeaseGroup()
        self._directory_fd_owner = _FDLeaseGroup()
        self._directory_cleared = False
        self._lock = threading.RLock()
        self._lock_pid = os.getpid()

    def _current_lock(self) -> threading.RLock:
        if self._lock_pid != os.getpid():
            # Never acquire a lock inherited from another PID: a vanished
            # parent thread may have held it across fork().
            self._lock = threading.RLock()
            self._lock_pid = os.getpid()
        return self._lock

    def adopt_directories(self, state: Mapping[str, object]) -> None:
        required = {
            "parent_fd",
            "parent_stat",
            "cache_fd",
            "cache_stat",
            "leaf",
            "owner_pid",
        }
        if set(state) != required:
            raise ValueError("invalid private-cache directory ownership state")
        published_state = dict(state)
        with self._current_lock():
            if self.released or self._directory is not None:
                raise ValueError("private-cache directories are already adopted")
            # Normal SIGINT is blocked while both directory leases and their
            # binding state move into this owner.
            with blocked_sigint():
                self._directory_fd_owner.adopt_fds(
                    (state["parent_fd"], state["cache_fd"])
                )
                self._directory = published_state

    def owns_directory_fd(self, fd: int) -> bool:
        with self._current_lock():
            return self._directory_fd_owner.owns_fd(fd)

    def _require_directory(self) -> dict[str, object]:
        if self._directory is None:
            raise ValueError("verified private HF cache directories are not adopted")
        return self._directory

    @property
    def dataset_pid(self) -> int | None:
        with self._current_lock():
            if self.released or self.phase[0] != "ready":
                return None
            return self.phase[1]

    @property
    def dataset(self) -> datasets.Dataset | None:
        with self._current_lock():
            if self.released or self.phase[0] != "ready":
                return None
            return self.phase[2]

    @property
    def fds(self) -> tuple[int, ...]:
        with self._current_lock():
            return self._arrow_fd_owner.fds

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.cache_fd}"

    @property
    def parent_fd(self) -> int:
        value = self._require_directory().get("parent_fd")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("invalid verified private-cache parent FD")
        return value

    @property
    def cache_fd(self) -> int:
        value = self._require_directory().get("cache_fd")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("invalid verified private-cache FD")
        return value

    @property
    def owner_pid(self) -> int | None:
        if self._directory is None:
            return None
        value = self._directory.get("owner_pid")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def directory_state(self) -> dict[str, object]:
        with self._current_lock():
            return dict(self._require_directory())

    @contextmanager
    def operation(self):
        # The lock deliberately spans build/mmap or one complete sample read.
        # A concurrent cleanup cannot close/reuse the actual FD mid-operation.
        with self._current_lock():
            if self.released:
                raise ValueError("verified private HF cache ownership is released")
            yield self

    def assert_directory_bound(self) -> None:
        with self._current_lock():
            if self.released:
                raise ValueError("verified private HF cache ownership is released")
            directory = self._require_directory()
            transient_owner = _FDLeaseGroup()
            try:
                with blocked_sigint():
                    verified_parent_fd, _ = _open_secure_absolute_directory(
                        self.parent_path,
                        create=False,
                    )
                    transient_owner.adopt_fd(verified_parent_fd)
                parent_path_binding = _directory_binding(
                    os.fstat(verified_parent_fd)
                )
            finally:
                transient_owner.release()
            leaf_path_stat = os.stat(
                directory["leaf"],
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            if (
                _directory_binding(os.fstat(self.parent_fd))
                != directory["parent_stat"]
                or parent_path_binding != directory["parent_stat"]
                or _directory_binding(os.fstat(self.cache_fd))
                != directory["cache_stat"]
                or _directory_binding(leaf_path_stat) != directory["cache_stat"]
                or not stat.S_ISDIR(leaf_path_stat.st_mode)
            ):
                raise ValueError("verified private HF cache directory binding changed")

    def adopt_arrow_entry(self, entry: Mapping[str, object]) -> None:
        logical_path = entry.get("logical_path")
        fd = entry.get("fd")
        if (
            not isinstance(logical_path, str)
            or not logical_path
            or isinstance(fd, bool)
            or not isinstance(fd, int)
        ):
            raise ValueError("invalid pinned Arrow ownership entry")
        with self._current_lock():
            if self.released or self.phase[0] != "building":
                raise ValueError("cannot adopt Arrow FD into inactive cache ownership")
            if logical_path in self.entries:
                raise ValueError("duplicate pinned Arrow ownership entry")
            updated = dict(self.entries)
            updated[logical_path] = dict(entry)
            with blocked_sigint():
                self._arrow_fd_owner.adopt_fd(fd)
                # One pointer store publishes the entry and its FD together.
                self.entries = updated

    def owns_arrow_fd(self, fd: int) -> bool:
        with self._current_lock():
            return self._arrow_fd_owner.owns_fd(fd)

    def mark_ready(self, dataset: datasets.Dataset, dataset_pid: int) -> None:
        with self._current_lock():
            if self.released or self.phase[0] != "building" or not self.entries:
                raise ValueError("cannot ready incomplete private Arrow ownership")
            self.phase = ("ready", dataset_pid, dataset)

    def entries_snapshot(self, *, allow_building: bool = False) -> dict[str, dict]:
        with self._current_lock():
            if self.released or (
                self.phase[0] != "ready"
                and not (allow_building and self.phase[0] == "building")
            ):
                raise ValueError("private Arrow ownership is not ready")
            return dict(self.entries)

    def release(self, *, delete_owned: bool) -> None:
        with self._current_lock():
            if self.released:
                return
            self._finish_release(delete_owned=delete_owned)

    def _finish_release(self, *, delete_owned: bool) -> None:
        first_error = None
        try:
            self._arrow_fd_owner.release()
        except BaseException as exc:
            if not self._arrow_fd_owner.released:
                raise
            first_error = exc
        # Once Arrow ownership has been consumed, normal SIGINT is deferred
        # across directory cleanup and the final state transition. Directory
        # FDs themselves remain in a resumable lease for ordinary exceptions.
        with blocked_sigint():
            self._release_directories(delete_owned=delete_owned)
            self.phase = ("released", None, None)
            self.released = True
        if first_error is not None:
            raise first_error

    def _release_directories(self, *, delete_owned: bool) -> None:
        if self._directory is None:
            self._directory_fd_owner.release()
            return
        directory = self._directory
        cache_fd = self.cache_fd
        parent_fd = self.parent_fd
        should_delete = delete_owned and directory["owner_pid"] == os.getpid()
        if should_delete and not self._directory_cleared:
            verified_parent_fd = None
            transient_owner = _FDLeaseGroup()
            try:
                with blocked_sigint():
                    verified_parent_fd, _ = _open_secure_absolute_directory(
                        self.parent_path, create=False
                    )
                    transient_owner.adopt_fd(verified_parent_fd)
                parent_path_binding = _directory_binding(
                    os.fstat(verified_parent_fd)
                )
                leaf_stat = os.stat(
                    directory["leaf"],
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    _directory_binding(os.fstat(parent_fd))
                    != directory["parent_stat"]
                    or parent_path_binding != directory["parent_stat"]
                    or _directory_binding(os.fstat(cache_fd))
                    != directory["cache_stat"]
                    or _directory_binding(leaf_stat) != directory["cache_stat"]
                    or not stat.S_ISDIR(leaf_stat.st_mode)
                ):
                    raise ValueError(
                        "verified private HF cache directory binding changed"
                    )
                _secure_clear_directory_fd(cache_fd)
                self._directory_cleared = True
            except Exception as exc:
                should_delete = False
                logging.error("cannot safely remove owned private HF cache: %s", exc)
            finally:
                transient_owner.release()

        first_error = None
        if should_delete and self._directory_fd_owner.owns_fd(cache_fd):
            try:
                cache_closed = self._directory_fd_owner.release_fd(cache_fd)
            except BaseException as exc:
                if self._directory_fd_owner.owns_fd(cache_fd):
                    raise
                first_error = exc
                cache_closed = True
            if not cache_closed:
                should_delete = False
                logging.error("cannot safely remove owned private HF cache: close failed")
        if should_delete:
            try:
                os.rmdir(directory["leaf"], dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logging.error("cannot safely remove owned private HF cache: %s", exc)
        try:
            self._directory_fd_owner.release()
        except BaseException as exc:
            if not self._directory_fd_owner.released:
                raise
            if first_error is None:
                first_error = exc
        self._directory = None
        self._directory_cleared = False
        if first_error is not None:
            raise first_error

    def __del__(self):
        try:
            self.release(delete_owned=True)
        except BaseException:
            pass


class _FDLeaseGroup:
    """Idempotent single owner for source FDs returned across Python calls."""

    def __init__(self) -> None:
        self._fds: list[int] = []
        self.released = False
        self._lock = threading.RLock()
        self._lock_pid = os.getpid()

    def _current_lock(self) -> threading.RLock:
        if self._lock_pid != os.getpid():
            self._lock = threading.RLock()
            self._lock_pid = os.getpid()
        return self._lock

    def adopt_fd(self, fd: int) -> None:
        self.adopt_fds((fd,))

    def adopt_fds(self, fds: Iterable[int]) -> None:
        candidates = tuple(fds)
        if not candidates or any(
            isinstance(fd, bool) or not isinstance(fd, int) for fd in candidates
        ):
            raise ValueError("invalid verified source FD")
        if len(set(candidates)) != len(candidates):
            raise ValueError("duplicate verified source FD")
        with self._current_lock():
            if self.released or any(fd in self._fds for fd in candidates):
                raise ValueError("cannot adopt source FD")
            with blocked_sigint():
                self._fds.extend(candidates)

    @property
    def fds(self) -> tuple[int, ...]:
        with self._current_lock():
            return tuple(self._fds)

    def owns_fd(self, fd: int) -> bool:
        with self._current_lock():
            return fd in self._fds

    def release_fd(self, fd: int) -> bool:
        """Consume one matching FD before its single close attempt."""

        with self._current_lock():
            if fd not in self._fds:
                return True
            with blocked_sigint():
                try:
                    os.close(self._fds.pop(self._fds.index(fd)))
                except OSError:
                    return False
            return True

    def relinquish_fd(self, fd: int) -> None:
        """Transfer an FD to a caller without closing it."""

        with self._current_lock():
            if fd not in self._fds:
                raise ValueError("cannot relinquish unowned verified FD")
            with blocked_sigint():
                self._fds.remove(fd)
                self.released = not self._fds

    @contextmanager
    def operation(self):
        # Source descriptors remain live through one complete sample read.
        # Cleanup takes this same lock only after the Arrow/cache read guard,
        # so video/parquet FD reuse cannot race a verified read.
        with self._current_lock():
            if self.released:
                raise ValueError("verified source FD ownership is released")
            yield self

    def release(self) -> None:
        with self._current_lock():
            if self.released:
                return
            first_error = None
            try:
                with blocked_sigint():
                    _close_fd_sequence_once(self._fds)
            except BaseException as exc:
                first_error = exc
            finally:
                self.released = not self._fds
            if first_error is not None:
                raise first_error

    def __del__(self):
        try:
            self.release()
        except BaseException:
            pass


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning(f'Invalid integer value for {name}={value!r}; using {default}.')
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


class _LeRobotVideoNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True

        if not record.pathname.replace('\\', '/').endswith('/lerobot/datasets/video_utils.py'):
            return True

        message = record.getMessage()
        return not (
            message.startswith('Using video codec:')
            or message.startswith('Auto-selected video codec:')
            or message.startswith('No hardware encoder available, falling back to software encoder')
        )


_LEROBOT_VIDEO_NOISE_FILTER_MARKER = '_giga_lerobot_video_noise_filter'


def _install_lerobot_log_filters() -> None:
    if not _get_env_bool('GIGA_LEROBOT_FILTER_NOISY_LOGS', True):
        return

    warnings.filterwarnings(
        'ignore',
        message=r'The video decoding and encoding capabilities of torchvision are deprecated.*',
        category=UserWarning,
        module=r'torchvision\.io\._video_deprecation_warning',
    )

    root_logger = logging.getLogger()
    if any(getattr(log_filter, _LEROBOT_VIDEO_NOISE_FILTER_MARKER, False) for log_filter in root_logger.filters):
        return

    log_filter = _LeRobotVideoNoiseFilter()
    setattr(log_filter, _LEROBOT_VIDEO_NOISE_FILTER_MARKER, True)
    root_logger.addFilter(log_filter)


_install_lerobot_log_filters()


def _is_hf_parquet_cache_read_error(error: BaseException) -> bool:
    visited: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))

        message = str(current)
        if (
            '.incomplete' in message
            and '-of-NNNNN.arrow' in message
            and ('No such file or directory' in message or 'Errno 2' in message)
        ):
            return True
        if 'Expected to be able to read' in message and 'for message body, got' in message:
            return True
        if 'Tried reading schema message, was null or length 0' in message:
            return True

        cause = getattr(current, '__cause__', None)
        context = getattr(current, '__context__', None)
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return False


def _has_complete_hf_arrow_cache(cache_path: Path) -> bool:
    info_path = cache_path / hf_datasets_config.DATASET_INFO_FILENAME
    if not info_path.is_file():
        return False

    try:
        info = DatasetInfo.from_directory(str(cache_path))
    except Exception as error:
        logging.warning(f'Failed to inspect HF parquet cache metadata {info_path}: {error}')
        return False

    if info.splits is None:
        return False

    for split_info in info.splits.values():
        filenames = filenames_for_dataset_split(
            str(cache_path),
            dataset_name='parquet',
            split=split_info.name,
            filetype_suffix='arrow',
            shard_lengths=split_info.shard_lengths,
        )
        if any(not Path(filename).is_file() for filename in filenames):
            return False
    return True


def _clear_hf_parquet_builder_cache(builder_cache_dir: str | None) -> bool:
    if builder_cache_dir is None:
        return False

    cache_path = Path(builder_cache_dir)
    lock_path = f'{builder_cache_dir}_builder.lock'
    timeout_s = max(1, _get_env_int('GIGA_LEROBOT_HF_CACHE_REPAIR_LOCK_TIMEOUT_S', 600))
    try:
        with FileLock(lock_path, timeout=timeout_s):
            if _has_complete_hf_arrow_cache(cache_path):
                return False

            cleared = False
            for path in (cache_path, Path(f'{builder_cache_dir}.incomplete')):
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                        cleared = True
                    else:
                        existed = path.exists()
                        path.unlink(missing_ok=True)
                        cleared = cleared or existed
                except OSError as error:
                    logging.warning(f'Failed to clear HF parquet cache path {path}: {error}')
            return cleared
    except Timeout:
        logging.warning(
            f'Timed out after {timeout_s}s waiting to repair HF parquet cache {builder_cache_dir}; '
            'another process may still be building it.'
        )
        return False


def _format_lerobot_context(
    root: str | Path | None,
    *,
    repo_id: str | None = None,
    episodes: Sequence[int] | None = None,
    cache_dir: str | None = None,
    builder_cache_dir: str | None = None,
    parquet_count: int | None = None,
) -> str:
    parts = [f'root={root!s}']
    if repo_id is not None:
        parts.append(f'repo_id={repo_id}')
    if episodes is not None:
        preview = list(episodes[:10]) if not isinstance(episodes, list) else episodes[:10]
        suffix = '' if len(episodes) <= 10 else f'... total={len(episodes)}'
        parts.append(f'episodes={preview}{suffix}')
    if cache_dir is not None:
        parts.append(f'cache_dir={cache_dir}')
    if builder_cache_dir is not None:
        parts.append(f'builder_cache_dir={builder_cache_dir}')
    if parquet_count is not None:
        parts.append(f'parquet_count={parquet_count}')
    parts.append(_get_data_worker_context())
    return ', '.join(parts)


def _load_lerobot_episodes_no_hf_cache(root: Path) -> datasets.Dataset:
    from lerobot.datasets.utils import EPISODES_DIR

    paths = sorted((root / EPISODES_DIR).glob('*/*.parquet'))
    if len(paths) == 0:
        raise FileNotFoundError(f'Provided directory does not contain any parquet file: {root / EPISODES_DIR}')

    tables = []
    for path in paths:
        table = pq.read_table(path)
        columns = [name for name in table.column_names if not name.startswith('stats/')]
        tables.append(table.select(columns))

    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options='default')
    return datasets.Dataset.from_dict(table.to_pydict())


def _get_lerobot_total_frames_from_info(root: Path) -> int:
    with (root / 'meta' / 'info.json').open('r') as f:
        return int(json.load(f)['total_frames'])


def _get_lerobot_episode_lengths_no_hf_cache(root: Path, episodes: Sequence[int]) -> int:
    legacy_path = root / 'meta' / 'episodes.jsonl'
    if legacy_path.is_file():
        lengths = {}
        with legacy_path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                lengths[int(row['episode_index'])] = int(row['length'])
        missing = sorted(set(int(index) for index in episodes) - set(lengths))
        if missing:
            raise IndexError(f'Episodes missing from {legacy_path}: {missing}')
        return sum(lengths[int(episode_idx)] for episode_idx in episodes)

    from lerobot.datasets.utils import EPISODES_DIR

    paths = sorted((root / EPISODES_DIR).glob('*/*.parquet'))
    if len(paths) == 0:
        raise FileNotFoundError(f'Provided directory does not contain any parquet file: {root / EPISODES_DIR}')

    columns = None
    episode_lengths: list[int] = []
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        if 'length' in schema_names:
            columns = ['length']
        elif {'dataset_from_index', 'dataset_to_index'}.issubset(schema_names):
            columns = ['dataset_from_index', 'dataset_to_index']
        else:
            raise KeyError(f'Missing episode length columns in {path}')

        table = pq.read_table(path, columns=columns)
        if 'length' in table.column_names:
            episode_lengths.extend(int(value) for value in table['length'].to_pylist())
        else:
            starts = table['dataset_from_index'].to_pylist()
            ends = table['dataset_to_index'].to_pylist()
            episode_lengths.extend(int(end) - int(start) for start, end in zip(starts, ends, strict=False))

    return sum(episode_lengths[int(episode_idx)] for episode_idx in episodes)


def _read_pinned_verified_bytes(entry: Mapping[str, object]) -> bytes:
    fd = int(entry["fd"])
    expected_stat = entry["stat"]
    before = os.fstat(fd)
    if stat_identity(before) != expected_stat:
        raise ValueError(f"pinned source stat changed: {entry['logical_path']}")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks = []
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        chunks.append(block)
    after = os.fstat(fd)
    if stat_identity(after) != expected_stat or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"pinned source changed while reading: {entry['logical_path']}")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _strict_pinned_json(entry: Mapping[str, object]) -> dict:
    payload = strict_json_loads(
        _read_pinned_verified_bytes(entry),
        name=str(entry["logical_path"]),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"verified metadata is not an object: {entry['logical_path']}")
    return payload


def _open_secure_absolute_directory(path: Path, *, create: bool) -> tuple[int, dict]:
    """Open an absolute directory without following a symlink component."""

    absolute = path.expanduser().absolute()
    if not absolute.is_absolute():
        raise ValueError("private HF cache parent must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd_owner = _FDLeaseGroup()
    current_fd = None
    try:
        with blocked_sigint():
            current_fd = os.open(os.path.sep, flags)
            fd_owner.adopt_fd(current_fd)
        for part in absolute.parts[1:]:
            try:
                with blocked_sigint():
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                    fd_owner.adopt_fd(next_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    # Another rank may have atomically created this shared
                    # parent component; the no-follow open below is still the
                    # authority and rejects non-directories/symlinks.
                    pass
                with blocked_sigint():
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                    fd_owner.adopt_fd(next_fd)
            with blocked_sigint():
                fd_owner.release_fd(current_fd)
                current_fd = next_fd
        runtime_stat = stat_identity(os.fstat(current_fd))
        if not stat.S_ISDIR(runtime_stat["mode"]):
            raise ValueError("private HF cache parent is not a directory")
        with blocked_sigint():
            fd_owner.relinquish_fd(current_fd)
        return current_fd, runtime_stat
    finally:
        fd_owner.release()


def _secure_clear_directory_fd(directory_fd: int) -> None:
    """Remove only entries reached through an already-pinned directory FD."""

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    for name in os.listdir(directory_fd):
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            child_owner = _FDLeaseGroup()
            try:
                with blocked_sigint():
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                    child_owner.adopt_fd(child_fd)
                _secure_clear_directory_fd(child_fd)
            finally:
                child_owner.release()
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _open_verified_regular_fd_at(
    root_fd: int,
    relative_path: str,
    *,
    cache_ownership: _CacheOwnership | None = None,
    arrow_entry_fields: Mapping[str, object] | None = None,
) -> tuple[int, str, dict] | dict:
    """Open/hash a regular descendant without following any path component."""

    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError("invalid private-cache relative path")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    file_owner = _FDLeaseGroup()
    directory_owner = _FDLeaseGroup()
    fd = None
    try:
        with blocked_sigint():
            directory_fd = os.dup(root_fd)
            directory_owner.adopt_fd(directory_fd)
        try:
            for part in relative.parts[:-1]:
                with blocked_sigint():
                    next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                    directory_owner.adopt_fd(next_fd)
                    directory_owner.release_fd(directory_fd)
                    directory_fd = next_fd
            name = relative.parts[-1]
            path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            with blocked_sigint():
                fd = os.open(name, file_flags, dir_fd=directory_fd)
                file_owner.adopt_fd(fd)
        finally:
            directory_owner.release()
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError("private Arrow cache entry is not a pinned regular file")
        digest = hashlib.sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(fd)
        if stat_identity(before) != stat_identity(after):
            raise ValueError("private Arrow cache changed while hashing")
        os.lseek(fd, 0, os.SEEK_SET)
        if cache_ownership is not None:
            if not isinstance(arrow_entry_fields, Mapping):
                raise ValueError("pinned Arrow ownership entry fields are missing")
            entry = {
                **arrow_entry_fields,
                "sha256": digest.hexdigest(),
                "stat": stat_identity(after),
                "fd": fd,
                "pinned_path": f"/proc/self/fd/{fd}",
            }
            with blocked_sigint():
                cache_ownership.adopt_arrow_entry(entry)
                file_owner.relinquish_fd(fd)
            return entry
        result = fd, digest.hexdigest(), stat_identity(after)
        with blocked_sigint():
            file_owner.relinquish_fd(fd)
        return result
    finally:
        directory_owner.release()
        file_owner.release()


def _stat_regular_at(root_fd: int, relative_path: str) -> dict:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError("invalid private-cache relative path")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_owner = _FDLeaseGroup()
    try:
        with blocked_sigint():
            directory_fd = os.dup(root_fd)
            directory_owner.adopt_fd(directory_fd)
        for part in relative.parts[:-1]:
            with blocked_sigint():
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                directory_owner.adopt_fd(next_fd)
                directory_owner.release_fd(directory_fd)
                directory_fd = next_fd
        path_stat = os.stat(
            relative.parts[-1],
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("private Arrow pathname is not a regular file")
        return stat_identity(path_stat)
    finally:
        directory_owner.release()


class _LeRobotDatasetMetadata(_OriginalLeRobotDatasetMetadata):
    def load_metadata(self):
        from lerobot.datasets.utils import (
            check_version_compatibility,
            load_info,
            load_stats,
            load_tasks,
        )

        verified = _VERIFIED_METADATA_CONTEXT.get()
        if verified is None:
            self.info = load_info(self.root)
        else:
            self.info = _strict_pinned_json(verified["meta/info.json"])
            for feature in self.info["features"].values():
                feature["shape"] = tuple(feature["shape"])
        if str(self.info.get('codebase_version', '')).lower() == 'v2.1':
            tasks_path = self.root / 'meta' / 'tasks.jsonl'
            episode_path = self.root / 'meta' / 'episodes.jsonl'
            task_rows = []
            if verified is None:
                task_lines = tasks_path.read_bytes().splitlines()
            else:
                task_lines = _read_pinned_verified_bytes(
                    verified["meta/tasks.jsonl"]
                ).splitlines()
            for line_number, line in enumerate(task_lines, 1):
                if line.strip():
                    task_rows.append(
                        strict_json_loads(
                            line,
                            name=f"meta/tasks.jsonl:{line_number}",
                        )
                    )
            task_rows.sort(key=lambda row: int(row['task_index']))
            self.tasks = pd.DataFrame(
                {'task_index': [int(row['task_index']) for row in task_rows]},
                index=[str(row['task']) for row in task_rows],
            )
            self.subtasks = None

            episode_rows = []
            dataset_from_index = 0
            if verified is None:
                episode_lines = episode_path.read_bytes().splitlines()
            else:
                episode_lines = _read_pinned_verified_bytes(
                    verified["meta/episodes.jsonl"]
                ).splitlines()
            for line_number, line in enumerate(episode_lines, 1):
                if line.strip():
                    episode_rows.append(
                        strict_json_loads(
                            line,
                            name=f"meta/episodes.jsonl:{line_number}",
                        )
                    )
            episode_rows.sort(key=lambda row: int(row['episode_index']))
            expected_indices = list(range(len(episode_rows)))
            actual_indices = [int(row['episode_index']) for row in episode_rows]
            if actual_indices != expected_indices:
                raise ValueError(
                    f'LeRobot v2.1 episode indices must be contiguous: {actual_indices[:20]}'
                )
            for row in episode_rows:
                length = int(row['length'])
                row['dataset_from_index'] = dataset_from_index
                dataset_from_index += length
                row['dataset_to_index'] = dataset_from_index
            if dataset_from_index != int(self.info['total_frames']):
                raise ValueError(
                    'LeRobot v2.1 episode lengths do not match info.total_frames: '
                    f'{dataset_from_index} != {self.info["total_frames"]}'
                )
            self.episodes = datasets.Dataset.from_list(episode_rows)
            logging.info('Loaded local LeRobot v2.1 metadata through the read-only compatibility path.')
        else:
            try:
                from lerobot.datasets.utils import load_subtasks
            except ImportError:
                load_subtasks = lambda _root: None
            check_version_compatibility(
                self.repo_id,
                self._version,
                _lerobot_dataset_module.CODEBASE_VERSION,
            )
            self.tasks = load_tasks(self.root)
            self.subtasks = load_subtasks(self.root)
            self.episodes = _load_lerobot_episodes_no_hf_cache(self.root)
        if verified is not None and "meta/stats.json" in verified:
            from lerobot.datasets.utils import cast_stats_to_numpy

            self.stats = cast_stats_to_numpy(
                _strict_pinned_json(verified["meta/stats.json"])
            )
        elif verified is not None:
            self.stats = None
        else:
            self.stats = load_stats(self.root)


_lerobot_dataset_module.LeRobotDatasetMetadata = _LeRobotDatasetMetadata


def _is_valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_delta_offsets(delta_name: str, delta_spec: TemporalOffsetSpec) -> list[int]:
    if _is_valid_int(delta_spec):
        if delta_spec < 0:
            raise ValueError(f"delta_info['{delta_name}'] must be non-negative when provided as an int.")
        return list(range(delta_spec))

    if isinstance(delta_spec, Sequence) and not isinstance(delta_spec, (str, bytes)):
        offsets: list[int] = []
        for offset in delta_spec:
            if not _is_valid_int(offset):
                raise TypeError(
                    f"delta_info['{delta_name}'] offsets must be integers, got {type(offset).__name__}."
                )
            offsets.append(offset)
        return offsets

    if not isinstance(delta_spec, Mapping):
        raise TypeError(
            f"delta_info['{delta_name}'] must be an int, a list[int], or a dict, got {type(delta_spec).__name__}."
        )

    has_offsets = 'offsets' in delta_spec
    has_window = any(key in delta_spec for key in ('start', 'stop', 'stride'))
    if has_offsets and has_window:
        raise ValueError(
            f"delta_info['{delta_name}'] cannot mix 'offsets' with 'start'/'stop'/'stride'."
        )

    if has_offsets:
        unknown_keys = set(delta_spec) - {'offsets'}
        if unknown_keys:
            raise ValueError(
                f"delta_info['{delta_name}'] has unsupported keys for offsets mode: {sorted(unknown_keys)}."
            )
        return _normalize_delta_offsets(delta_name, delta_spec['offsets'])

    unknown_keys = set(delta_spec) - {'start', 'stop', 'stride'}
    if unknown_keys:
        raise ValueError(
            f"delta_info['{delta_name}'] has unsupported keys for range mode: {sorted(unknown_keys)}."
        )
    if 'start' not in delta_spec or 'stop' not in delta_spec:
        raise ValueError(f"delta_info['{delta_name}'] range mode requires both 'start' and 'stop'.")

    start = delta_spec['start']
    stop = delta_spec['stop']
    stride = delta_spec.get('stride', 1)
    if not _is_valid_int(start) or not _is_valid_int(stop) or not _is_valid_int(stride):
        raise TypeError(
            f"delta_info['{delta_name}'] range mode expects integer 'start', 'stop', and 'stride'."
        )
    if stride <= 0:
        raise ValueError(f"delta_info['{delta_name}'] stride must be a positive integer.")
    return list(range(start, stop, stride))


def _build_delta_timestamps(
    delta_info: dict[str, TemporalOffsetSpec] | None,
    fps: int,
) -> dict[str, list[float]] | None:
    if delta_info is None:
        return None

    delta_timestamps: dict[str, list[float]] = {}
    for delta_name, delta_spec in delta_info.items():
        delta_timestamps[delta_name] = [offset / fps for offset in _normalize_delta_offsets(delta_name, delta_spec)]
    return delta_timestamps


def _normalize_video_key_spec(video_keys: VideoKeySpec | None, *, arg_name: str) -> tuple[str, ...] | None:
    if video_keys is None:
        return None
    if isinstance(video_keys, str):
        return (video_keys,)
    if not isinstance(video_keys, Iterable):
        raise TypeError(f'{arg_name} should be a string, a sequence of strings, or None.')

    normalized = []
    seen = set()
    for key in video_keys:
        if not isinstance(key, str):
            raise TypeError(f'{arg_name} should contain strings, got {type(key).__name__}.')
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    return tuple(normalized)


def _select_nearest_frame(
    previous_frame: tuple[float, torch.Tensor] | None,
    current_frame: tuple[float, torch.Tensor] | None,
    query_timestamp: float,
) -> tuple[float, torch.Tensor]:
    if previous_frame is None and current_frame is None:
        raise FrameTimestampError(f'No frame could be decoded for timestamp {query_timestamp}.')
    if previous_frame is None:
        return current_frame
    if current_frame is None:
        return previous_frame

    previous_distance = abs(previous_frame[0] - query_timestamp)
    current_distance = abs(current_frame[0] - query_timestamp)
    if previous_distance <= current_distance:
        return previous_frame
    return current_frame


def _raise_frame_tolerance_error(
    video_path: Path | str,
    timestamps: list[float],
    query_timestamp: float,
    matched_timestamp: float,
    tolerance_s: float,
) -> None:
    raise FrameTimestampError(
        f'Closest frame violates tolerance ({abs(matched_timestamp - query_timestamp)} >= {tolerance_s=}).'
        ' It means that the closest frame that can be loaded from the video is too far away in time.'
        ' This might be due to synchronization issues with timestamps during data collection.'
        ' To be safe, we advise to ignore this item during training.'
        f'\nqueried timestamps: {torch.tensor(timestamps)}'
        f'\nclosest timestamp: {matched_timestamp}'
        f'\nvideo: {video_path}'
        '\nbackend: pyav'
    )


def _decode_video_frames_pyav_streaming(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode timestamps with bounded CPU working set for the pyav backend.

    Unlike torchvision's default implementation, this keeps at most the previous
    and current decoded frames while scanning forward from the nearest keyframe.
    """
    if not timestamps:
        raise ValueError('timestamps must not be empty.')

    sorted_query_positions = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    sorted_timestamps = [timestamps[pos] for pos in sorted_query_positions]

    torchvision.set_video_backend('pyav')
    reader = torchvision.io.VideoReader(str(video_path), 'video')
    reader.seek(sorted_timestamps[0], keyframes_only=True)

    selected_frames: list[torch.Tensor | None] = [None] * len(timestamps)
    selected_timestamps: list[float | None] = [None] * len(timestamps)
    previous_frame: tuple[float, torch.Tensor] | None = None
    query_cursor = 0

    try:
        for frame in reader:
            current_timestamp = float(frame['pts'])
            current_frame = (current_timestamp, frame['data'])

            if log_loaded_timestamps:
                logging.info(f'frame loaded at timestamp={current_timestamp:.4f}')

            while query_cursor < len(sorted_timestamps) and current_timestamp >= sorted_timestamps[query_cursor]:
                matched_timestamp, matched_frame = _select_nearest_frame(
                    previous_frame,
                    current_frame,
                    sorted_timestamps[query_cursor],
                )
                if abs(matched_timestamp - sorted_timestamps[query_cursor]) >= tolerance_s:
                    _raise_frame_tolerance_error(
                        video_path,
                        timestamps,
                        sorted_timestamps[query_cursor],
                        matched_timestamp,
                        tolerance_s,
                    )

                original_position = sorted_query_positions[query_cursor]
                selected_frames[original_position] = matched_frame
                selected_timestamps[original_position] = matched_timestamp
                query_cursor += 1

            previous_frame = current_frame
            if query_cursor == len(sorted_timestamps):
                break
    finally:
        container = getattr(reader, 'container', None)
        if container is not None:
            container.close()
        reader = None

    if previous_frame is None:
        raise FrameTimestampError(f'No frames could be decoded from video: {video_path}')

    while query_cursor < len(sorted_timestamps):
        matched_timestamp, matched_frame = previous_frame
        if abs(matched_timestamp - sorted_timestamps[query_cursor]) >= tolerance_s:
            _raise_frame_tolerance_error(
                video_path,
                timestamps,
                sorted_timestamps[query_cursor],
                matched_timestamp,
                tolerance_s,
            )

        original_position = sorted_query_positions[query_cursor]
        selected_frames[original_position] = matched_frame
        selected_timestamps[original_position] = matched_timestamp
        query_cursor += 1

    closest_frames = torch.stack([frame for frame in selected_frames if frame is not None])
    closest_frames = closest_frames.type(torch.float32) / 255

    if log_loaded_timestamps:
        logging.info(f'closest_timestamps={torch.tensor(selected_timestamps)}')

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f'Number of retrieved frames ({len(closest_frames)}) does not match '
            f'number of queried timestamps ({len(timestamps)})'
        )
    return closest_frames


@register_dataset
class LeRobotDataset(BaseDataset):
    """Wrapper around lerobot's dataset for unified interface with our
    BaseDataset.

    Supports optional ``delta_info`` to compute additional timestamp offsets and
    attaching dataset-level meta under ``meta_name`` in each returned sample.
    ``delta_info`` values can be legacy integers, explicit signed offset lists,
    or structured ``{'start', 'stop', 'stride'}`` windows.
    """

    def __init__(
        self,
        data_path: str,
        data_size: int | None = None,
        delta_info: dict[str, TemporalOffsetSpec] | None = None,
        meta_name: str | None = None,
        repack_transform: dict | None = None,
        decode_video_keys: VideoKeySpec | None = None,
        **kwargs,
    ) -> None:
        super(LeRobotDataset, self).__init__(data_path=data_path)
        self.data_size = data_size
        self.delta_info = delta_info
        self.meta_name = meta_name
        self.repack_transform = repack_transform
        self.decode_video_keys = _normalize_video_key_spec(decode_video_keys, arg_name='decode_video_keys')
        self.kwargs = kwargs
        self.dataset = None

    def _load_metadata(self) -> _LeRobotDatasetMetadata:
        repo_id = os.path.basename(os.path.normpath(self.data_path))
        return _LeRobotDatasetMetadata(repo_id, root=self.data_path)

    def _get_metadata_data_size(self) -> int:
        root = Path(self.data_path)
        episodes = self.kwargs.get('episodes')
        if episodes is None:
            return _get_lerobot_total_frames_from_info(root)
        return _get_lerobot_episode_lengths_no_hf_cache(root, episodes)

    @classmethod
    def load(cls, data_or_config: str | dict) -> 'LeRobotDataset':
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def open(self) -> None:
        if self.dataset is None:
            try:
                # Build metadata and optionally derive delta timestamps per stream
                dataset_meta = self._load_metadata()
                delta_timestamps = _build_delta_timestamps(self.delta_info, dataset_meta.fps)
                # Use faster subclass that avoids temporary disk writes during conversion
                self.dataset = FastLeRobotDataset(
                    dataset_meta.repo_id,
                    root=self.data_path,
                    delta_timestamps=delta_timestamps,
                    video_backend='pyav',
                    repack_transform=self.repack_transform,
                    decode_video_keys=self.decode_video_keys,
                    **self.kwargs,
                )
                if self.data_size is not None:
                    assert self.data_size == len(self.dataset)
                else:
                    self.data_size = len(self.dataset)
            except Exception as error:
                raise RuntimeError(
                    'Failed to open LeRobotDataset: '
                    f'{_format_lerobot_context(self.data_path, episodes=self.kwargs.get("episodes"))}, '
                    f'delta_keys={list(self.delta_info or {})}, kwargs_keys={list(self.kwargs)}'
                ) from error

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset = None
        super(LeRobotDataset, self).close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.data_size = self._get_metadata_data_size()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        data_dict = self.dataset[index]
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            # Attach global meta under user-provided key
            data_dict[self.meta_name] = self.dataset.meta
        return data_dict


class FastLeRobotDataset(_LeRobotDataset):
    """This class overrides the `LeRobotDataset`(lerobot version 0.4.4) class
    to accelerate the data conversion process.

    What it does is:
    - Doesn't store temporary image files to disk, instead, it's kept in memory until the whole episode is saved.
    - Only consider observation.state and action features to compute episode statistics.

    Beside, it's recommended to use video mode rather than image mode when converting large datasets. It's easy for data transfer and storage.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 2e-2,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
        skip_video_decoding: bool = False,
        repack_transform: dict | None = None,
        decode_video_keys: VideoKeySpec | None = None,
        verified_source_receipt: list[dict] | None = None,
        verified_private_cache_parent: str | Path | None = None,
        verified_training_run_nonce: str | None = None,
    ):
        self._verified_source_entries: dict[str, dict] = {}
        self._verified_source_fd_owner = _FDLeaseGroup()
        self._verified_parquet_paths: list[Path] | None = None
        self._verified_video_entries: dict[tuple[int, str], dict] = {}
        self._verified_cache_tag: str | None = None
        self._verified_pid = os.getpid()
        self._verified_cache_ownership: _CacheOwnership | None = None
        self._verified_cache_coordinator = (os.getpid(), threading.RLock())
        self._verified_private_cache_parent = (
            None
            if verified_private_cache_parent is None
            else Path(verified_private_cache_parent).expanduser().absolute()
        )
        self._verified_training_run_nonce = verified_training_run_nonce
        token = None
        try:
            if verified_source_receipt is not None:
                if (
                    self._verified_private_cache_parent is None
                    or not isinstance(self._verified_training_run_nonce, str)
                    or not _TRAINING_NONCE_RE.fullmatch(
                        self._verified_training_run_nonce
                    )
                ):
                    raise ValueError(
                        "verified source loading requires an explicit private-cache parent "
                        "and training run nonce"
                    )
                self._prepare_verified_sources(root, verified_source_receipt)
                metadata_entries = {
                    logical: entry
                    for logical, entry in self._verified_source_entries.items()
                    if entry["logical_role"] == "metadata"
                }
                token = _VERIFIED_METADATA_CONTEXT.set(metadata_entries)
            super().__init__(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                revision=revision,
                force_cache_sync=force_cache_sync,
                download_videos=download_videos,
                video_backend=video_backend,
            )
            self._assert_verified_sources_unchanged()
        except BaseException as exc:
            self.close_verified_sources()
            if _is_file_descriptor_exhaustion(exc):
                raise ValueError(
                    "insufficient file-descriptor capacity for verified LeRobot sources/cache"
                ) from exc
            raise
        finally:
            if token is not None:
                _VERIFIED_METADATA_CONTEXT.reset(token)
        self.skip_video_decoding = skip_video_decoding
        self.repack_transform = repack_transform
        self.decode_video_keys = _normalize_video_key_spec(decode_video_keys, arg_name='decode_video_keys')
        self._decode_video_key_set: set[str] | None = None
        self._repack_mode = 'none'
        self._repack_pairs = ()
        self._repack_assignments = ()
        # Tracks (src_key, dst_key) pairs whose source has already been reported
        # as missing for this dataset instance, so we warn once per pair instead
        # of spamming the log on every __getitem__.
        self._repack_warned_missing: set[tuple[str, str]] = set()
        if repack_transform:
            has_nested = any(isinstance(value, dict) for value in repack_transform.values())
            if has_nested:
                # Structured mode: map source flat keys into target nested structure.
                self._repack_mode = 'structured'
                self._repack_assignments = self._build_repack_assignments(repack_transform)
            else:
                # Flat mode: support both src->dst and dst->src styles at runtime.
                self._repack_mode = 'flat'
                self._repack_pairs = self._build_repack_pairs(repack_transform)

    def _prepare_verified_sources(
        self, root: str | Path | None, receipt: list[dict]
    ) -> None:
        if root is None or not isinstance(receipt, list) or not receipt:
            raise ValueError("verified LeRobot source receipt is empty")
        source_root = Path(root).expanduser().absolute()
        if source_root.is_symlink() or not source_root.is_dir():
            raise ValueError("verified LeRobot root must be a real directory")
        for raw_entry in receipt:
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
                "logical_path", "logical_role", "path_record_only", "sha256", "stat"
            }:
                raise ValueError("invalid verified LeRobot source receipt entry")
            entry = dict(raw_entry)
            logical = entry["logical_path"]
            role = entry["logical_role"]
            receipt_stat = entry["stat"]
            if (
                not isinstance(logical, str)
                or not logical
                or Path(logical).is_absolute()
                or ".." in Path(logical).parts
                or Path(logical).as_posix() != logical
                or logical in self._verified_source_entries
                or role not in {"metadata", "parquet", "video"}
                or not isinstance(entry["path_record_only"], str)
                or not Path(entry["path_record_only"]).is_absolute()
                or not isinstance(entry["sha256"], str)
                or not _SHA256_RE.fullmatch(entry["sha256"])
                or not isinstance(receipt_stat, Mapping)
                or set(receipt_stat) != _VERIFIED_STAT_KEYS
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in receipt_stat.values()
                )
            ):
                raise ValueError("invalid/duplicate verified LeRobot logical source")
            actual_path = Path(entry["path_record_only"]).expanduser().absolute()
            if role in {"metadata", "parquet"}:
                expected_path = (source_root / logical).absolute()
                if actual_path != expected_path:
                    raise ValueError(f"verified source path differs for {logical}")
            fd, digest, runtime_stat = open_verified_regular_fd(
                actual_path,
                expected_sha256=entry["sha256"],
                expected_stat=entry["stat"],
                fd_owner=self._verified_source_fd_owner,
            )
            entry.update(
                fd=fd,
                sha256=digest,
                stat=runtime_stat,
                pinned_path=Path(f"/proc/self/fd/{fd}"),
            )
            self._verified_source_entries[logical] = entry
        required_metadata = {
            "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"
        }
        receipt_metadata = {
            entry["logical_path"]
            for entry in self._verified_source_entries.values()
            if entry["logical_role"] == "metadata"
        }
        if receipt_metadata not in (
            required_metadata,
            required_metadata | {"meta/stats.json"},
        ):
            raise ValueError("verified LeRobot metadata receipt is incomplete")
        actual_parquet = {
            path.absolute() for path in (source_root / "data").glob("*/*.parquet")
        }
        receipt_parquet = {
            Path(entry["path_record_only"]).absolute()
            for entry in self._verified_source_entries.values()
            if entry["logical_role"] == "parquet"
        }
        if not receipt_parquet or receipt_parquet != actual_parquet:
            raise ValueError("verified LeRobot parquet set is missing or has extras")
        parquet_entries = sorted(
            (
                entry
                for entry in self._verified_source_entries.values()
                if entry["logical_role"] == "parquet"
            ),
            key=lambda entry: entry["logical_path"],
        )
        self._verified_parquet_paths = [entry["pinned_path"] for entry in parquet_entries]
        for entry in self._verified_source_entries.values():
            if entry["logical_role"] != "video":
                continue
            match = re.fullmatch(r"videos/(.+)/episode_([0-9]{6})\.mp4", entry["logical_path"])
            if match is None:
                raise ValueError("invalid verified video logical path")
            key = (int(match.group(2)), match.group(1))
            if key in self._verified_video_entries:
                raise ValueError("duplicate verified video source")
            self._verified_video_entries[key] = entry
        tag_payload = [
            (entry["logical_path"], entry["logical_role"], entry["sha256"])
            for entry in sorted(
                self._verified_source_entries.values(),
                key=lambda value: value["logical_path"],
            )
        ]
        self._verified_cache_tag = hashlib.sha256(
            json.dumps(tag_payload, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()

    def _assert_verified_sources_unchanged(self) -> None:
        if self._verified_source_entries and self._verified_pid != os.getpid():
            inherited_fds_valid = True
            for entry in self._verified_source_entries.values():
                try:
                    inherited_fds_valid = (
                        stat_identity(os.fstat(entry["fd"])) == entry["stat"]
                    )
                except (OSError, TypeError):
                    inherited_fds_valid = False
                if not inherited_fds_valid:
                    break
            if inherited_fds_valid:
                # fork() preserves the already-hashed open file descriptions;
                # no pathname reopen or multi-GB rehash is needed.
                self._verified_pid = os.getpid()
            else:
                # A spawn/pickle worker has no inherited descriptors. Never
                # close the serialized integer values: the child may have
                # independently reused those numbers for unrelated resources.
                old_owner = getattr(self, "_verified_source_fd_owner", None)
                if isinstance(old_owner, _FDLeaseGroup):
                    old_owner.release()
                reopened_owner = _FDLeaseGroup()
                try:
                    reopened_entries = {}
                    for logical, old_entry in self._verified_source_entries.items():
                        entry = dict(old_entry)
                        new_fd, _, _ = open_verified_regular_fd(
                            entry["path_record_only"],
                            expected_sha256=entry["sha256"],
                            expected_stat=entry["stat"],
                            fd_owner=reopened_owner,
                        )
                        entry["fd"] = new_fd
                        entry["pinned_path"] = Path(f"/proc/self/fd/{new_fd}")
                        reopened_entries[logical] = entry
                except BaseException:
                    reopened_owner.release()
                    raise
                reopened_parquet_paths = [
                    entry["pinned_path"]
                    for entry in sorted(
                        (
                            value
                            for value in reopened_entries.values()
                            if value["logical_role"] == "parquet"
                        ),
                        key=lambda value: value["logical_path"],
                    )
                ]
                reopened_video_entries = {
                    key: reopened_entries[entry["logical_path"]]
                    for key, entry in self._verified_video_entries.items()
                }
                # This is the only source-state replacement. Normal SIGINT is
                # blocked so the descriptor owner cannot be separated from the
                # mappings that reference it within the supported model.
                with blocked_sigint():
                    self._verified_source_entries = reopened_entries
                    self._verified_parquet_paths = reopened_parquet_paths
                    self._verified_video_entries = reopened_video_entries
                    self._verified_source_fd_owner = reopened_owner
                    self._verified_pid = os.getpid()
        for entry in self._verified_source_entries.values():
            if stat_identity(os.fstat(entry["fd"])) != entry["stat"]:
                raise ValueError(
                    f"verified LeRobot source changed during loader open: {entry['logical_path']}"
                )

    def _ensure_verified_process(self) -> None:
        with self._cache_coordinator_lock():
            if self._verified_source_entries and self._verified_pid != os.getpid():
                self._assert_verified_sources_unchanged()
            cache_ownership = getattr(self, "_verified_cache_ownership", None)
            dataset_pid = (
                cache_ownership.dataset_pid
                if isinstance(cache_ownership, _CacheOwnership)
                else None
            )
            if self._verified_source_entries and dataset_pid != os.getpid():
                # A fork/spawn worker must never consume the creating process's
                # private cache or inherited Arrow mmap.  It retains/reopens only
                # the verified source FDs, then force-builds its own unpredictable
                # cache and maps the newly pinned Arrow descriptors.
                self._release_private_cache(delete_owned=False)
                self.hf_dataset = None
                self.hf_dataset = self.load_hf_dataset()

    def _cache_coordinator_lock(self) -> threading.RLock:
        coordinator = getattr(self, "_verified_cache_coordinator", None)
        if (
            not isinstance(coordinator, tuple)
            or len(coordinator) != 2
            or coordinator[0] != os.getpid()
        ):
            # Check PID before touching an inherited lock: another parent
            # thread may have held it across fork().
            coordinator = (os.getpid(), threading.RLock())
            self._verified_cache_coordinator = coordinator
        return coordinator[1]

    @staticmethod
    def _assert_verified_entry_unchanged(entry: Mapping[str, object]) -> None:
        if stat_identity(os.fstat(entry["fd"])) != entry["stat"]:
            raise ValueError(
                f"verified LeRobot source changed: {entry['logical_path']}"
            )

    def _validate_verified_embedded_images(self, dataset: datasets.Dataset) -> None:
        image_keys = [
            key
            for key, feature in self.features.items()
            if feature.get("dtype") == "image"
        ]
        for key in image_keys:
            column = dataset.data.column(key)
            for chunk in column.chunks:
                if not pa.types.is_struct(chunk.type) or "bytes" not in (
                    field.name for field in chunk.type
                ):
                    raise ValueError(
                        f"verified image column is not an embedded bytes struct: {key}"
                    )
                embedded = chunk.field("bytes")
                if embedded.null_count or bool(
                    pc.any(pc.equal(pc.binary_length(embedded), 0)).as_py()
                ):
                    raise ValueError(
                        f"verified image column contains path-only/empty images: {key}"
                    )

    def _create_private_cache(self) -> _CacheOwnership:
        if self._verified_private_cache_parent is None:
            raise ValueError("verified private-cache parent is missing")
        existing = getattr(self, "_verified_cache_ownership", None)
        if isinstance(existing, _CacheOwnership) and not existing.released:
            raise ValueError("verified private cache is already open")
        parent_fd = None
        cache_fd = None
        leaf = None
        ownership = _CacheOwnership(parent_path=self._verified_private_cache_parent)
        try:
            with blocked_sigint():
                parent_fd, _ = _open_secure_absolute_directory(
                    self._verified_private_cache_parent,
                    create=True,
                )
            parent_stat = os.fstat(parent_fd)
            if (
                parent_stat.st_uid != os.geteuid()
                or stat.S_IMODE(parent_stat.st_mode) != 0o700
            ):
                raise ValueError(
                    "verified private-cache parent must be owned by this user and mode 0700"
                )
            rank_text = os.environ.get("RANK", "unranked")
            if rank_text != "unranked" and not re.fullmatch(
                r"[0-9]+", rank_text, flags=re.ASCII
            ):
                raise ValueError(
                    "RANK must be canonical ASCII decimal for private cache"
                )
            nonce_tag = hashlib.sha256(
                self._verified_training_run_nonce.encode("utf-8")
            ).hexdigest()[:16]
            for _ in range(32):
                candidate = (
                    f"run-{nonce_tag}-rank-{rank_text}-pid-{os.getpid()}-"
                    f"{secrets.token_hex(16)}"
                )
                leaf = candidate
                try:
                    os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    leaf = None
                    continue
                break
            if leaf is None:
                raise ValueError("cannot allocate unpredictable private HF cache")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            with blocked_sigint():
                cache_fd = os.open(leaf, flags, dir_fd=parent_fd)
            os.fchmod(cache_fd, 0o700)
            cache_stat = os.fstat(cache_fd)
            if (
                cache_stat.st_uid != os.geteuid()
                or stat.S_IMODE(cache_stat.st_mode) != 0o700
            ):
                raise ValueError("private HF cache is not owned mode 0700")
            ownership.adopt_directories(
                {
                    "parent_fd": parent_fd,
                    "parent_stat": _directory_binding(os.fstat(parent_fd)),
                    "cache_fd": cache_fd,
                    "cache_stat": _directory_binding(cache_stat),
                    "leaf": leaf,
                    "owner_pid": os.getpid(),
                }
            )
            ownership.assert_directory_bound()
            return ownership
        except BaseException:
            if (
                (parent_fd is not None and ownership.owns_directory_fd(parent_fd))
                or (cache_fd is not None and ownership.owns_directory_fd(cache_fd))
            ):
                ownership.release(delete_owned=True)
            else:
                try:
                    if cache_fd is not None:
                        _secure_clear_directory_fd(cache_fd)
                    if leaf is not None and parent_fd is not None:
                        os.rmdir(leaf, dir_fd=parent_fd)
                except OSError:
                    pass
                finally:
                    with blocked_sigint():
                        try:
                            if cache_fd is not None:
                                old_fd, cache_fd = cache_fd, None
                                _close_fd_once(old_fd)
                        finally:
                            if parent_fd is not None:
                                old_fd, parent_fd = parent_fd, None
                                _close_fd_once(old_fd)
            raise

    def _assert_private_cache_directory_bound(self) -> None:
        ownership = getattr(self, "_verified_cache_ownership", None)
        if not isinstance(ownership, _CacheOwnership):
            raise ValueError("verified private HF cache is not pinned")
        ownership.assert_directory_bound()

    def _open_pinned_arrow_dataset(
        self,
        reader,
        builder_cache_dir: Path,
        ownership: _CacheOwnership,
    ) -> datasets.Dataset:
        try:
            with ownership.operation():
                split_name = str(reader.split)
                split_info = reader.builder.info.splits[split_name]
                arrow_paths = [
                    Path(path)
                    for path in filenames_for_dataset_split(
                        str(builder_cache_dir),
                        dataset_name=reader.builder.dataset_name,
                        split=split_name,
                        filetype_suffix="arrow",
                        shard_lengths=split_info.shard_lengths,
                    )
                ]
                if not arrow_paths:
                    raise ValueError("private HF build produced no Arrow files")
                parts = []
                private_root = ownership.proc_path
                for index, arrow_path in enumerate(arrow_paths):
                    relative_path = Path(
                        os.path.relpath(str(arrow_path), private_root)
                    ).as_posix()
                    entry = _open_verified_regular_fd_at(
                        ownership.cache_fd,
                        relative_path,
                        cache_ownership=ownership,
                        arrow_entry_fields={
                            "logical_path": f"arrow/{index:06d}",
                            "relative_path": relative_path,
                            "build_path_record_only": str(arrow_path),
                        },
                    )
                    if not isinstance(entry, dict):
                        raise ValueError("pinned Arrow ownership adoption failed")
                    fd = entry["fd"]
                    before = stat_identity(os.fstat(fd))
                    part = datasets.Dataset.from_file(
                        entry["pinned_path"],
                        info=reader.builder.info,
                        split=reader.split,
                        in_memory=False,
                    )
                    if stat_identity(os.fstat(fd)) != before:
                        raise ValueError("private Arrow inode changed while mmap opened")
                    parts.append(part)
                result = (
                    parts[0]
                    if len(parts) == 1
                    else datasets.concatenate_datasets(
                        parts,
                        info=reader.builder.info,
                        split=reader.split,
                    )
                )
                actual_mmap_paths = {
                    item.get("filename") for item in result.cache_files
                }
                expected_mmap_paths = {
                    entry["pinned_path"] for entry in ownership.entries.values()
                }
                if actual_mmap_paths != expected_mmap_paths:
                    raise ValueError(
                        "HF Dataset is not mmap-backed by the pinned private Arrow FDs"
                    )
                return result
        except BaseException:
            ownership.release(delete_owned=True)
            raise

    def _assert_verified_cache_unchanged(
        self, ownership: _CacheOwnership | None = None
    ) -> None:
        if not self._verified_source_entries:
            return
        if ownership is None:
            ownership = getattr(self, "_verified_cache_ownership", None)
        if not isinstance(ownership, _CacheOwnership):
            raise ValueError("verified private HF cache is not pinned")
        with ownership.operation():
            ownership.assert_directory_bound()
            for entry in ownership.entries_snapshot(allow_building=True).values():
                if (
                    stat_identity(os.fstat(entry["fd"])) != entry["stat"]
                    or _stat_regular_at(
                        ownership.cache_fd,
                        entry["relative_path"],
                    )
                    != entry["stat"]
                ):
                    raise ValueError(
                        "pinned private Arrow mmap source changed during training read"
                    )

    def __getstate__(self):
        with self._cache_coordinator_lock():
            state = dict(self.__dict__)
            source_entries = {
                logical: {**entry, "fd": None, "pinned_path": None}
                for logical, entry in self._verified_source_entries.items()
            }
            state["_verified_source_entries"] = source_entries
            state["_verified_source_fd_owner"] = None
            state["_verified_parquet_paths"] = None
            state["_verified_video_entries"] = {
                key: source_entries[entry["logical_path"]]
                for key, entry in self._verified_video_entries.items()
            }
            state["_verified_pid"] = None
            state["hf_dataset"] = None
            state["_verified_cache_ownership"] = None
            state["_verified_cache_coordinator"] = None
            return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._verified_cache_coordinator = (os.getpid(), threading.RLock())

    def close_verified_sources(self) -> None:
        # Serialize cleanup with private-cache construction/publication.  Cache
        # release goes first: it waits for any complete sample read before the
        # source descriptors used by that read can be closed and reused.
        with self._cache_coordinator_lock():
            source_owner = getattr(self, "_verified_source_fd_owner", None)
            try:
                self._release_private_cache(delete_owned=True)
            finally:
                if isinstance(source_owner, _FDLeaseGroup):
                    source_owner.release()

    def _release_private_cache(self, *, delete_owned: bool) -> None:
        # RLock makes this safe when the caller already owns the coordinator
        # (normal close and build replacement paths).
        with self._cache_coordinator_lock():
            ownership = getattr(self, "_verified_cache_ownership", None)
            if isinstance(ownership, _CacheOwnership):
                owned_dataset = ownership.dataset
                try:
                    ownership.release(delete_owned=delete_owned)
                finally:
                    if getattr(self, "hf_dataset", None) is owned_dataset:
                        self.hf_dataset = None
                    owned_dataset = None

    def __del__(self):
        self.close_verified_sources()

    def _load_verified_hf_dataset(
        self,
        *,
        features,
        filters,
        parquet_paths: list[str],
        hf_transform,
    ) -> datasets.Dataset:
        from datasets.io.parquet import ParquetDatasetReader

        with self._cache_coordinator_lock():
            existing = getattr(self, "_verified_cache_ownership", None)
            if isinstance(existing, _CacheOwnership):
                existing_dataset = existing.dataset
                if existing.dataset_pid == os.getpid() and existing_dataset is not None:
                    return existing_dataset
                existing.release(delete_owned=existing.owner_pid == os.getpid())

            ownership = None
            try:
                ownership = self._create_private_cache()
                builder_identity_payload = {
                    "source_manifest_content_sha256": self._verified_cache_tag,
                    "columns": list(features),
                    "features": features.to_dict(),
                    "episodes": None
                    if self.episodes is None
                    else list(self.episodes),
                }
                builder_identity = hashlib.sha256(
                    json.dumps(
                        builder_identity_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                ownership.assert_directory_bound()
                self._assert_verified_sources_unchanged()
                reader = ParquetDatasetReader(
                    parquet_paths,
                    filters=filters,
                    features=features,
                    columns=list(features),
                    cache_dir=ownership.proc_path,
                )
                reader.builder = _VerifiedParquetBuilder(
                    cache_dir=ownership.proc_path,
                    dataset_name="parquet",
                    data_files={str(reader.split): parquet_paths},
                    features=features,
                    hash=reader.builder.hash,
                    filters=filters,
                    columns=list(features),
                    verified_identity=builder_identity,
                )
                builder_cache_dir = Path(reader.builder.cache_dir)
                # Every verified load rebuilds inside this unpublished,
                # unpredictable owner; no shared cache can authorize bytes.
                reader.builder.download_and_prepare(
                    download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD,
                )
                ownership.assert_directory_bound()
                self._assert_verified_sources_unchanged()
                hf_dataset = self._open_pinned_arrow_dataset(
                    reader,
                    builder_cache_dir,
                    ownership,
                )
                self._assert_verified_cache_unchanged(ownership)
                self._validate_verified_embedded_images(hf_dataset)
                self._assert_verified_cache_unchanged(ownership)
                hf_dataset.set_transform(hf_transform)
                ownership.mark_ready(hf_dataset, os.getpid())
                # The only live-cache publication: directory bindings, Arrow
                # entries/FDs, Dataset, PID and READY phase move together.
                self._verified_cache_ownership = ownership
                return hf_dataset
            except BaseException as exc:
                if isinstance(ownership, _CacheOwnership):
                    ownership.release(delete_owned=True)
                if isinstance(exc, Exception):
                    if _is_file_descriptor_exhaustion(exc):
                        raise ValueError(
                            "insufficient file-descriptor capacity for pinned Arrow cache"
                        ) from exc
                    raise ValueError(
                        "verified private HF build/mmap admission failed"
                    ) from exc
                raise

    @override
    def load_hf_dataset(self) -> datasets.Dataset:
        """Load parquet data using the metadata-declared columns only.

        Some converted LeRobot shards contain extra parquet columns that are not
        present in ``meta/info.json``. HuggingFace Datasets raises a CastError
        when casting such files with an explicit features schema. Restricting the
        parquet read to the declared feature columns preserves LeRobot's
        metadata contract and ignores unused extra columns.
        """
        from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
        from datasets.io.parquet import ParquetDatasetReader
        from pyarrow import dataset as pa_ds

        features = get_hf_features_from_features(self.features)
        paths = self._verified_parquet_paths or sorted(
            (self.root / 'data').glob('*/*.parquet')
        )
        if len(paths) == 0:
            raise FileNotFoundError(f'Provided directory does not contain any parquet file: {self.root / "data"}')

        filters = pa_ds.field('episode_index').isin(self.episodes) if self.episodes is not None else None
        parquet_paths = [str(path) for path in paths]
        if self._verified_cache_tag is not None:
            return self._load_verified_hf_dataset(
                features=features,
                filters=filters,
                parquet_paths=parquet_paths,
                hf_transform=hf_transform_to_torch,
            )

        cache_dir = os.environ.get('HF_DATASETS_CACHE') or os.path.expanduser('~/.cache/huggingface/datasets')
        builder_cache_dir: str | None = None

        def load_from_parquet() -> datasets.Dataset:
            nonlocal builder_cache_dir
            reader = ParquetDatasetReader(
                parquet_paths,
                filters=filters,
                features=features,
                columns=list(features),
                cache_dir=cache_dir,
            )
            builder_cache_dir = reader.builder.cache_dir
            return reader.read()

        def load_from_parquet_with_retry() -> datasets.Dataset:
            attempts = max(1, _get_env_int('GIGA_LEROBOT_HF_CACHE_LOAD_ATTEMPTS', 10))
            for attempt in range(attempts):
                try:
                    return load_from_parquet()
                except Exception as error:
                    is_recoverable_cache_error = (
                        self._verified_cache_tag is None
                        and _is_hf_parquet_cache_read_error(error)
                    )
                    if not is_recoverable_cache_error:
                        raise
                    if attempt + 1 == attempts:
                        raise RuntimeError(
                            'HF parquet cache remained missing or truncated after '
                            f'{attempts} attempts: '
                            f'{_format_lerobot_context(self.root, repo_id=self.repo_id, episodes=self.episodes, cache_dir=cache_dir, builder_cache_dir=builder_cache_dir, parquet_count=len(parquet_paths))}'
                        ) from error
                    cleared_cache = _clear_hf_parquet_builder_cache(builder_cache_dir)
                    delay_s = min(2.0, 0.2 * (2**attempt))
                    logging.warning(
                        'HF parquet cache is missing or truncated; '
                        f'{"cleared" if cleared_cache else "kept"} {builder_cache_dir} and retrying in {delay_s:.1f}s '
                        f'({attempt + 1}/{attempts}). '
                        f'{_format_lerobot_context(self.root, repo_id=self.repo_id, episodes=self.episodes, cache_dir=cache_dir, builder_cache_dir=builder_cache_dir, parquet_count=len(parquet_paths))}'
                    )
                    time.sleep(delay_s)
            raise RuntimeError('Unreachable HF parquet cache retry state.')

        if self._verified_cache_tag is None and _get_env_bool('GIGA_LEROBOT_LOCK_HF_CACHE', True):
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            if cache_path.is_symlink() or not cache_path.is_dir():
                raise ValueError("verified HF cache namespace must be a real directory")
            lock_path = cache_path / 'giga_lerobot_from_parquet.lock'
            with FileLock(str(lock_path)):
                hf_dataset = load_from_parquet_with_retry()
        elif self._verified_cache_tag is None:
            hf_dataset = load_from_parquet_with_retry()

        if self._verified_source_entries:
            self._assert_verified_cache_unchanged()
            self._validate_verified_embedded_images(hf_dataset)
            self._assert_verified_cache_unchanged()
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @override
    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """This function only adds the frame to the episode_buffer and nothing
        is written to disk.

        To save those frames, the 'save_episode()' method then needs to be called.
        """
        # Convert torch tensors to numpy arrays for serialization/storage
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer['size']
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer['frame_index'].append(frame_index)
        self.episode_buffer['timestamp'].append(timestamp)
        self.episode_buffer['task'].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")

            self.episode_buffer[key].append(frame[key])

        self.episode_buffer['size'] += 1

    @override
    def save_episode(self, episode_data: dict | None = None) -> None:
        """This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # 'size' and 'task' are bookkeeping fields, omitted from parquet payload
        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length,), episode_index)

        # Register any new tasks encountered during this episode.
        self.meta.save_episode_tasks(episode_tasks)

        # Map natural-language task names to task indices
        episode_buffer['task_index'] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ['index', 'episode_index', 'task_index'] or ft['dtype'] in ['image', 'video']:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = _compute_episode_stats(episode_buffer)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_buffer, episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # Persist episode meta after encoding videos to include video metadata.
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, {})

        video_files = list(self.root.rglob('*.mp4'))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob('*.parquet'))
        assert len(parquet_files) == self.num_episodes

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    @override
    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split='train')
        ep_dataset = embed_images(ep_dataset)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    @override
    def encode_episode_videos(self, episode_buffer: dict, episode_index: int) -> dict:
        """Use ffmpeg to convert frames stored as png into mp4 videos.

        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue

            imgs = episode_buffer[key]

            _encode_video_frames(imgs, video_path, self.fps, overwrite=True)

        return video_paths

    @override
    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        ep = self.meta.episodes[ep_idx]
        item = {}
        video_backend = self.video_backend or get_safe_default_codec()
        query_timestamps = self._filter_query_timestamps(query_timestamps)
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f'videos/{vid_key}/from_timestamp']
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            verified_video = self._verified_video_entries.get((ep_idx, vid_key))
            if self._verified_source_entries and verified_video is None:
                raise ValueError(
                    f"missing verified video receipt for episode={ep_idx}, key={vid_key}"
                )
            video_path = (
                verified_video["pinned_path"]
                if verified_video is not None
                else self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            )
            if verified_video is not None:
                self._assert_verified_entry_unchanged(verified_video)
            if video_backend == 'pyav':
                frames = _decode_video_frames_pyav_streaming(video_path, shifted_query_ts, self.tolerance_s)
            else:
                frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, video_backend)
            if verified_video is not None:
                self._assert_verified_entry_unchanged(verified_video)
            item[vid_key] = frames.squeeze(0)

        return item

    @override
    def __getitem__(self, idx: int) -> dict:
        guards = ExitStack()
        try:
            with self._cache_coordinator_lock():
                self._ensure_verified_process()
                self._ensure_hf_dataset_loaded()
                source_owner = getattr(self, "_verified_source_fd_owner", None)
                ownership = getattr(self, "_verified_cache_ownership", None)
                # Enter both read guards before dropping the coordinator. Thus
                # cleanup cannot observe a just-returned build and race either
                # hf_dataset assignment or the first verified sample read.
                if isinstance(source_owner, _FDLeaseGroup):
                    guards.enter_context(source_owner.operation())
                if isinstance(ownership, _CacheOwnership):
                    guards.enter_context(ownership.operation())
            return self._getitem_from_loaded_dataset(idx)
        finally:
            guards.close()

    def _getitem_from_loaded_dataset(self, idx: int) -> dict:
        self._assert_verified_cache_unchanged()
        item = self.hf_dataset[idx]
        ep_idx = item['episode_index'].item()
        abs_idx = item['index'].item()

        query_indices = None
        if self.delta_indices is not None:
            # lerobot>=0.4 expects absolute dataset index here.
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Optional: skip costly video decoding when only computing stats
        if not self.skip_video_decoding and len(self.meta.video_keys) > 0:
            current_ts = item['timestamp'].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                if self._verified_source_entries:
                    raise
                logging.warning(
                    f'Failed to decode video frames for episode {ep_idx} in timestamps: {query_timestamps}. Error: {e}. Falling back to zeros.'
                )
                video_frames = {}
                # Construct zero tensors matching expected shapes per key
                for vid_key, query_ts in self._filter_query_timestamps(query_timestamps).items():
                    num_queries = len(query_ts)
                    # Prefer shapes from metadata when available
                    ft_shape = self.meta.shapes.get(vid_key)

                    # Derive channel-first (C,H,W)
                    if isinstance(ft_shape, tuple) and len(ft_shape) == 3:
                        if ft_shape[0] in (1, 3, 4):  # likely CHW
                            c, h, w = ft_shape[0], ft_shape[1], ft_shape[2]
                        elif ft_shape[2] in (1, 3, 4):  # likely HWC
                            c, h, w = ft_shape[2], ft_shape[0], ft_shape[1]
                        else:
                            c, h, w = 3, ft_shape[0], ft_shape[1]
                    else:
                        # Conservative default
                        c, h, w = 3, 224, 224

                    if num_queries > 1:
                        zeros_shape = (num_queries, c, h, w)
                    else:
                        zeros_shape = (c, h, w)

                    video_frames[vid_key] = torch.zeros(zeros_shape, dtype=torch.float32)

            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item['task_index'].item()
        item['task'] = self.meta.tasks.iloc[task_idx].name
        if self._repack_mode != 'none':
            item = self._apply_repack_transform(item)
        self._assert_verified_cache_unchanged()
        return item

    def _filter_query_timestamps(self, query_timestamps: dict[str, list[float]]) -> dict[str, list[float]]:
        decode_video_key_set = self._get_decode_video_key_set()
        if decode_video_key_set is None:
            return query_timestamps
        return {key: value for key, value in query_timestamps.items() if key in decode_video_key_set}

    def _get_decode_video_key_set(self) -> set[str] | None:
        if self.decode_video_keys is None:
            return None
        if self._decode_video_key_set is None:
            available_keys = set(self.meta.video_keys)
            resolved_keys: set[str] = set()
            missing_keys = []
            for requested_key in self.decode_video_keys:
                candidates = self._candidate_source_keys_for_decode_key(requested_key)
                source_key = next((key for key in candidates if key in available_keys), None)
                if source_key is None:
                    missing_keys.append(requested_key)
                    continue
                resolved_keys.add(source_key)

            if missing_keys:
                print(
                    f'[decode_video_keys] dataset {self.root!s}: requested keys {missing_keys!r} '
                    f'not present in video keys {sorted(available_keys)!r}; skipping them.',
                    flush=True,
                )
            self._decode_video_key_set = resolved_keys
        return self._decode_video_key_set

    def _candidate_source_keys_for_decode_key(self, requested_key: str) -> tuple[str, ...]:
        candidates = [requested_key]
        if self._repack_mode == 'flat':
            for left_key, right_key in self._repack_pairs:
                if left_key == requested_key and right_key not in candidates:
                    candidates.append(right_key)
                if right_key == requested_key and left_key not in candidates:
                    candidates.append(left_key)
        elif self._repack_mode == 'structured':
            for dst_path, src_key in self._repack_assignments:
                dst_key = '.'.join(dst_path)
                if dst_key == requested_key and src_key not in candidates:
                    candidates.append(src_key)
                if len(dst_path) == 1 and dst_path[0] == requested_key and src_key not in candidates:
                    candidates.append(src_key)
        return tuple(candidates)

    @staticmethod
    def _build_repack_pairs(repack_transform: dict[str, str]) -> tuple[tuple[str, str], ...]:
        pairs = [(src_key, dst_key) for src_key, dst_key in repack_transform.items() if src_key != dst_key]
        explicit_src_keys = {src_key for src_key, _ in pairs}
        explicit_dst_keys = {dst_key for _, dst_key in pairs}

        for src_key, dst_key in tuple(pairs):
            if src_key.endswith('_is_pad') or dst_key.endswith('_is_pad'):
                continue

            src_pad_key = f'{src_key}_is_pad'
            dst_pad_key = f'{dst_key}_is_pad'
            if src_pad_key in explicit_src_keys or dst_pad_key in explicit_dst_keys:
                continue
            if src_pad_key == dst_pad_key:
                continue

            pairs.append((src_pad_key, dst_pad_key))

        return tuple(pairs)

    @staticmethod
    def _collect_repack_leaf_mappings(repack_transform: dict) -> tuple[tuple[tuple[str, ...], str], ...]:
        assignments: list[tuple[tuple[str, ...], str]] = []

        def _walk(node: dict, prefix: tuple[str, ...] = ()) -> None:
            for dst_key, mapping in node.items():
                dst_path = prefix + (dst_key,)
                if isinstance(mapping, dict):
                    _walk(mapping, dst_path)
                    continue
                if not isinstance(mapping, str):
                    raise TypeError(
                        f'Invalid repack_transform leaf type for key {dst_path}. '
                        f'Expected str or dict, got {type(mapping).__name__}.'
                    )
                assignments.append((dst_path, mapping))

        _walk(repack_transform)
        return tuple(assignments)

    @staticmethod
    def _build_repack_assignments(repack_transform: dict) -> tuple[tuple[tuple[str, ...], str], ...]:
        explicit_mappings = FastLeRobotDataset._collect_repack_leaf_mappings(repack_transform)
        explicit_src_keys = {src_key for _, src_key in explicit_mappings}
        explicit_dst_paths = {dst_path for dst_path, _ in explicit_mappings}
        assignments: list[tuple[tuple[str, ...], str]] = []

        for dst_path, src_key in explicit_mappings:
            # No-op mapping (e.g. {'action': 'action'}) can be skipped.
            if len(dst_path) != 1 or dst_path[0] != src_key:
                assignments.append((dst_path, src_key))

            if src_key.endswith('_is_pad') or dst_path[-1].endswith('_is_pad'):
                continue

            src_pad_key = f'{src_key}_is_pad'
            dst_pad_path = dst_path[:-1] + (f'{dst_path[-1]}_is_pad',)
            if src_pad_key in explicit_src_keys or dst_pad_path in explicit_dst_paths:
                continue
            if len(dst_pad_path) == 1 and dst_pad_path[0] == src_pad_key:
                continue

            assignments.append((dst_pad_path, src_pad_key))

        return tuple(assignments)

    def _warn_repack_miss(self, src_key: str, dst_key: str) -> None:
        """Log once per (src, dst) pair that the configured source key is absent.

        Dataset groups in giga-brain configs share a single repack_transform,
        but individual paths may legitimately lack some optional cameras. We
        skip those mappings silently for training, but emit a single warning
        per dataset instance + key pair so the path can be located.
        """
        if src_key.endswith('_is_pad') or dst_key.endswith('_is_pad'):
            # Auto-derived pad pairs follow their parent; warning on the parent
            # is enough — don't double-log.
            return
        pair = (src_key, dst_key)
        if pair in self._repack_warned_missing:
            return
        self._repack_warned_missing.add(pair)
        print(
            f'[repack_transform] dataset {self.root!s}: source key {src_key!r} '
            f'(target {dst_key!r}) not present in sample; skipping this mapping.',
            flush=True,
        )

    def _apply_repack_transform(self, item: dict) -> dict:
        # Rename/repack in place to avoid allocating a new dict per sample.
        if self._repack_mode == 'flat':
            for left_key, right_key in self._repack_pairs:
                left_exists = left_key in item
                right_exists = right_key in item

                if left_exists and not right_exists:
                    src_key, dst_key = left_key, right_key
                elif right_exists and not left_exists:
                    src_key, dst_key = right_key, left_key
                elif left_exists and right_exists:
                    # Ambiguous case: keep backward-compatible behavior (left -> right).
                    src_key, dst_key = left_key, right_key
                else:
                    self._warn_repack_miss(left_key, right_key)
                    continue

                item[dst_key] = item.pop(src_key)

            return item

        if self._repack_mode == 'structured':
            for dst_path, src_key in self._repack_assignments:
                if src_key not in item:
                    self._warn_repack_miss(src_key, '.'.join(dst_path))
                    continue

                value = item.pop(src_key)

                if len(dst_path) == 1:
                    item[dst_path[0]] = value
                    continue

                parent = item
                for path_key in dst_path[:-1]:
                    child = parent.get(path_key)
                    if not isinstance(child, dict):
                        child = {}
                        parent[path_key] = child
                    parent = child
                parent[dst_path[-1]] = value

        return item


def _encode_video_frames(
    imgs: list[np.ndarray],
    video_path: Path | str,
    fps: int,
    vcodec: str = 'libsvtav1',
    pix_fmt: str = 'yuv420p',
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """Encode a sequence of RGB frames into a video file using PyAV/ffmpeg.

    Args:
        imgs (list[np.ndarray]): List of frames as HxWx3 RGB numpy arrays. All frames
            are assumed to share the same spatial size.
        video_path (Path | str): Output path for the encoded video file.
        fps (int): Target frames per second for the output video.
        vcodec (str): Video codec passed to ffmpeg. Supported: {'h264', 'hevc', 'libsvtav1'}.
        pix_fmt (str): Pixel format for the encoder (e.g., 'yuv420p', 'yuv444p').
        g (int | None): GOP size (distance between keyframes). ``None`` keeps encoder default.
        crf (int | None): Constant Rate Factor controlling quality/bitrate (lower is higher quality).
            ``None`` keeps encoder default.
        fast_decode (int): Enable fast-decode tuning when non-zero (codec-dependent behavior).
        log_level (int | None): LibAV logging level (e.g., ``av.logging.ERROR``). ``None`` keeps default.
        overwrite (bool): If True, create parent directories as needed and allow overwriting.

    Raises:
        ValueError: If ``vcodec`` is unsupported.
        FileNotFoundError: If no frames are provided.
        OSError: If encoding appears to succeed but the output file is missing.

    Notes:
        More details about ffmpeg argument tuning can be found in `benchmark/video/README.md`.
    """
    # Check encoder availability
    if vcodec not in ['h264', 'hevc', 'libsvtav1']:
        raise ValueError(f'Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.')

    video_path = Path(video_path)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == 'libsvtav1' or vcodec == 'hevc') and pix_fmt == 'yuv444p':
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = 'yuv420p'

    # Define video output frame size (assuming all input frames are the same size)
    if len(imgs) == 0:
        raise FileNotFoundError('No images found.')
    dummy_image = Image.fromarray(imgs[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options['g'] = str(g)

    if crf is not None:
        video_options['crf'] = str(crf)

    if fast_decode:
        key = 'svtav1-params' if vcodec == 'libsvtav1' else 'tune'
        value = f'fast-decode={fast_decode}' if vcodec == 'libsvtav1' else 'fastdecode'
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger('libav').setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), 'w') as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in imgs:
            input_image = Image.fromarray(input_data).convert('RGB')
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f'Video encoding did not work. File not found: {video_path}.')


def _get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        'min': np.min(array, axis=axis, keepdims=keepdims),
        'max': np.max(array, axis=axis, keepdims=keepdims),
        'mean': np.mean(array, axis=axis, keepdims=keepdims),
        'std': np.std(array, axis=axis, keepdims=keepdims),
        'count': np.array([len(array)]),
    }


def _compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray]) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in ['observation.state', 'action']:
            continue

        ep_ft_array = data  # data is already a np.ndarray
        axes_to_reduce = 0  # compute stats over the first axis
        keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = _get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

    return ep_stats
