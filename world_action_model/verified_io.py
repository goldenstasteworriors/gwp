"""Single-open, no-follow file verification helpers for production artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedIOError(f"invalid strict JSON: {name}") from exc


def verified_head_code_manifest(
    repository_root: str | os.PathLike[str],
    logical_paths: tuple[str, ...],
    *,
    expected_commit: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compare runtime code bytes directly with bytes read from one pinned HEAD."""

    root = _absolute_no_resolve(repository_root)
    if root.is_symlink() or not root.is_dir():
        raise VerifiedIOError("code repository root must be a real directory")
    if not logical_paths or len(set(logical_paths)) != len(logical_paths):
        raise VerifiedIOError("code manifest paths must be non-empty and unique")
    for logical_path in logical_paths:
        path = Path(logical_path)
        if (
            not isinstance(logical_path, str)
            or not logical_path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != logical_path
        ):
            raise VerifiedIOError("code manifest contains an unsafe logical path")

    def git_bytes(*arguments: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VerifiedIOError("cannot read current HEAD code blob bytes") from exc
        return result.stdout

    commit = git_bytes("rev-parse", "--verify", "HEAD^{commit}").decode(
        "ascii"
    ).strip()
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
        or (expected_commit is not None and commit != expected_commit)
    ):
        raise VerifiedIOError("current HEAD commit differs from expected code identity")

    rows = []
    for logical_path in logical_paths:
        object_name = f"{commit}:{logical_path}"
        blob_sha1 = git_bytes("rev-parse", "--verify", object_name).decode(
            "ascii"
        ).strip()
        if len(blob_sha1) != 40 or any(
            char not in "0123456789abcdef" for char in blob_sha1
        ):
            raise VerifiedIOError("invalid current-HEAD code blob identity")
        head_payload = git_bytes("cat-file", "blob", object_name)
        head_sha256 = hashlib.sha256(head_payload).hexdigest()
        runtime_sha256, runtime_stat = hash_verified_regular_file(
            root / logical_path
        )
        if (
            head_sha256 != runtime_sha256
            or len(head_payload) != runtime_stat["size_bytes"]
        ):
            raise VerifiedIOError(
                f"runtime code bytes differ from current HEAD blob: {logical_path}"
            )
        rows.append(
            {
                "logical_path": logical_path,
                "git_blob_sha1": blob_sha1,
                "head_blob_sha256": head_sha256,
                "head_blob_size_bytes": len(head_payload),
                "runtime_sha256": runtime_sha256,
                "runtime_size_bytes": runtime_stat["size_bytes"],
            }
        )

    commit_after = git_bytes("rev-parse", "--verify", "HEAD^{commit}").decode(
        "ascii"
    ).strip()
    if commit_after != commit:
        raise VerifiedIOError("HEAD changed while reading code blob bytes")
    return commit, rows


def load_verified_head_python_module(
    repository_root: str | os.PathLike[str],
    logical_path: str,
    module_name: str,
    *,
    expected_commit: str,
    expected_row: Mapping[str, Any],
):
    """Compile and execute the exact verified source bytes for one HEAD file."""

    if module_name in sys.modules:
        raise VerifiedIOError(f"verified module name is already loaded: {module_name}")
    commit, rows = verified_head_code_manifest(
        repository_root,
        (logical_path,),
        expected_commit=expected_commit,
    )
    if commit != expected_commit or rows != [dict(expected_row)]:
        raise VerifiedIOError("verified module differs from expected code manifest")
    root = _absolute_no_resolve(repository_root)
    source_path = root / logical_path
    payload, digest, runtime_stat = read_verified_bytes(
        source_path,
        expected_sha256=expected_row["runtime_sha256"],
    )
    if (
        digest != expected_row["runtime_sha256"]
        or digest != expected_row["head_blob_sha256"]
        or len(payload) != expected_row["runtime_size_bytes"]
        or len(payload) != expected_row["head_blob_size_bytes"]
        or runtime_stat["size_bytes"] != expected_row["runtime_size_bytes"]
    ):
        raise VerifiedIOError("verified module source bytes differ")
    try:
        code = compile(payload, str(source_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise VerifiedIOError("cannot compile verified module source bytes") from exc
    spec = importlib.util.spec_from_loader(
        module_name,
        loader=None,
        origin=str(source_path),
    )
    if spec is None:
        raise VerifiedIOError("cannot create verified module spec")
    spec.has_location = True
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if (
            not isinstance(origin, str)
            or not os.path.samefile(module.__file__, source_path)
            or not os.path.samefile(origin, source_path)
        ):
            raise VerifiedIOError("verified module origin differs from manifest path")
        _, after_rows = verified_head_code_manifest(
            root,
            (logical_path,),
            expected_commit=expected_commit,
        )
        if after_rows != [dict(expected_row)]:
            raise VerifiedIOError("verified module source changed during execution")
        return module
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise


def assert_python_module_origin(module: Any, expected_path: str | os.PathLike[str]) -> None:
    """Require both module file and import spec origin to name one real file."""

    expected = _absolute_no_resolve(expected_path)
    module_file = getattr(module, "__file__", None)
    spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    try:
        matches = (
            isinstance(module_file, str)
            and isinstance(spec_origin, str)
            and os.path.samefile(module_file, expected)
            and os.path.samefile(spec_origin, expected)
        )
    except OSError as exc:
        raise VerifiedIOError("cannot verify project module origin") from exc
    if not matches:
        raise VerifiedIOError(f"project module origin differs: {expected}")


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
