"""Single-open, no-follow file verification helpers for production artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
from contextlib import contextmanager
from pathlib import Path
import stat
from typing import Any, Mapping


class VerifiedIOError(ValueError):
    pass


@contextmanager
def blocked_sigint():
    """Block SIGINT in this POSIX thread during a short FD transfer."""

    if not hasattr(signal, "pthread_sigmask"):
        raise RuntimeError("verified FD ownership requires POSIX pthread_sigmask")
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def strict_json_loads(payload: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedIOError(f"invalid strict JSON: {name}") from exc


def stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _absolute_no_resolve(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _open_regular_fd(
    path: str | os.PathLike[str],
    *,
    expected_stat: Mapping[str, int] | None = None,
    fd_owner: Any | None = None,
) -> tuple[int, Path, dict[str, int]]:

    absolute = _absolute_no_resolve(path)
    try:
        before_open = os.lstat(absolute)
    except OSError as exc:
        raise VerifiedIOError(f"cannot lstat verified source: {absolute}") from exc
    if stat.S_ISLNK(before_open.st_mode) or not stat.S_ISREG(before_open.st_mode):
        raise VerifiedIOError(f"verified source must be a non-symlink regular file: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = None
    try:
        with blocked_sigint():
            try:
                fd = os.open(absolute, flags)
            except OSError as exc:
                raise VerifiedIOError(
                    f"cannot safely open verified source: {absolute}"
                ) from exc
        before = os.fstat(fd)
        if stat_identity(before) != stat_identity(before_open):
            raise VerifiedIOError(f"verified source changed during open: {absolute}")
        if expected_stat is not None and stat_identity(before) != dict(expected_stat):
            raise VerifiedIOError(f"verified source stat receipt differs: {absolute}")
        result_fd = fd
        if fd_owner is not None:
            with blocked_sigint():
                fd_owner.adopt_fd(fd)
                fd = None
        return result_fd, absolute, stat_identity(before)
    except BaseException:
        if fd is not None and (
            fd_owner is None or not fd_owner.owns_fd(fd)
        ):
            with blocked_sigint():
                old_fd, fd = fd, None
                try:
                    os.close(old_fd)
                except OSError:
                    pass
        raise


def _read_and_hash_fd(
    fd: int,
    *,
    absolute: Path,
    identity: Mapping[str, int],
    capture_bytes: bool,
) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks = []
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if capture_bytes:
                chunks.append(block)
        after = os.fstat(fd)
        if stat_identity(after) != dict(identity):
            raise VerifiedIOError(f"verified source changed while reading: {absolute}")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.lseek(fd, 0, os.SEEK_SET)


def open_verified_regular_fd(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_stat: Mapping[str, int] | None = None,
    fd_owner: Any | None = None,
) -> tuple[int, str, dict[str, int]]:
    """Open once with no-follow, hash that fd, and return it pinned at offset 0."""

    fd = None
    try:
        with blocked_sigint():
            fd, absolute, identity = _open_regular_fd(
                path,
                expected_stat=expected_stat,
                fd_owner=fd_owner,
            )
        _, actual_sha256 = _read_and_hash_fd(
            fd,
            absolute=absolute,
            identity=identity,
            capture_bytes=False,
        )
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise VerifiedIOError(f"verified source content SHA differs: {absolute}")
        return fd, actual_sha256, identity
    except BaseException:
        if fd is not None and (
            fd_owner is None or not fd_owner.owns_fd(fd)
        ):
            with blocked_sigint():
                old_fd, fd = fd, None
                try:
                    os.close(old_fd)
                except OSError:
                    pass
        raise


def hash_verified_regular_file(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_stat: Mapping[str, int] | None = None,
) -> tuple[str, dict[str, int]]:
    fd = None
    try:
        with blocked_sigint():
            fd, digest, identity = open_verified_regular_fd(
                path,
                expected_sha256=expected_sha256,
                expected_stat=expected_stat,
            )
        return digest, identity
    finally:
        if fd is not None:
            with blocked_sigint():
                old_fd, fd = fd, None
                try:
                    os.close(old_fd)
                except OSError:
                    pass


def read_verified_bytes(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_stat: Mapping[str, int] | None = None,
) -> tuple[bytes, str, dict[str, int]]:
    fd = None
    try:
        with blocked_sigint():
            fd, absolute, identity = _open_regular_fd(
                path, expected_stat=expected_stat
            )
        payload, digest = _read_and_hash_fd(
            fd,
            absolute=absolute,
            identity=identity,
            capture_bytes=True,
        )
        if expected_sha256 is not None and digest != expected_sha256:
            raise VerifiedIOError(f"verified source content SHA differs: {absolute}")
        return payload, digest, identity
    finally:
        if fd is not None:
            with blocked_sigint():
                old_fd, fd = fd, None
                try:
                    os.close(old_fd)
                except OSError:
                    pass
