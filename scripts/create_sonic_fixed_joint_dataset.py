#!/usr/bin/env python3
"""Create a versioned SONIC dataset with selected joints fixed to constants.

The source dataset is never modified. Parquet data files are rewritten, while
large immutable assets are hard-linked when the source and destination share a
filesystem. Action supervision is regenerated from the corrected future
``observation.state`` within each episode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
EPISODE_COLUMN = "episode_index"
ALIGNMENT_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-joint",
        action="append",
        required=True,
        metavar="NAME=VALUE",
        help="Joint and fixed value; repeat for multiple joints.",
    )
    return parser.parse_args()


def parse_fixed_joints(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate joint name {name!r}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"Fixed value must be finite, got {raw_value!r}")
        result[name] = value
    return result


def load_joint_names(source: Path) -> tuple[list[str], dict]:
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info["features"]
    state_names = list(features[STATE_COLUMN]["names"])
    action_names = list(features[ACTION_COLUMN]["names"])
    if state_names != action_names:
        raise ValueError("State and action joint names differ")
    return state_names, info


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing column {name!r}")
    field = table.schema.field(index)
    array = pa.array(values.tolist(), type=field.type)
    return table.set_column(index, field, array)


def future_state_targets(states: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    targets = np.empty_like(states)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        if rows.size == 0:
            continue
        if rows.size > 1 and not np.all(np.diff(rows) == 1):
            raise ValueError(f"Episode {int(episode)} is not contiguous within its parquet file")
        targets[rows[:-1]] = states[rows[1:]]
        targets[rows[-1]] = states[rows[-1]]
    return targets


def parquet_compression(source_file: Path) -> str:
    metadata = pq.ParquetFile(source_file).metadata
    if metadata.num_row_groups == 0 or metadata.num_columns == 0:
        return "zstd"
    return str(metadata.row_group(0).column(0).compression).lower()


def rewrite_parquet(
    source_file: Path,
    output_file: Path,
    fixed_indices: dict[int, float],
) -> dict[str, float | int]:
    table = pq.read_table(source_file)
    for required in (STATE_COLUMN, ACTION_COLUMN, EPISODE_COLUMN):
        if required not in table.column_names:
            raise KeyError(f"{source_file} is missing {required!r}")

    states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
    source_actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
    episodes = np.asarray(table[EPISODE_COLUMN].to_pylist(), dtype=np.int64).reshape(-1)
    expected_source = future_state_targets(states, episodes)
    source_alignment_error = float(np.max(np.abs(source_actions - expected_source)))
    if source_alignment_error > ALIGNMENT_ATOL:
        raise ValueError(
            f"Source supervision is not future observation.state in {source_file}: "
            f"max_error={source_alignment_error}"
        )

    corrected_states = states.copy()
    for index, value in fixed_indices.items():
        corrected_states[:, index] = value
    corrected_actions = future_state_targets(corrected_states, episodes)

    table = replace_column(table, STATE_COLUMN, corrected_states)
    table = replace_column(table, ACTION_COLUMN, corrected_actions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        output_file,
        compression=parquet_compression(source_file),
        use_dictionary=True,
    )

    alignment_error = float(
        np.max(np.abs(corrected_actions - future_state_targets(corrected_states, episodes)))
    )
    fixed_error = max(
        float(np.max(np.abs(corrected_states[:, index] - value)))
        for index, value in fixed_indices.items()
    )
    return {
        "rows": int(len(table)),
        "episodes": int(len(np.unique(episodes))),
        "source_alignment_max_abs_error": source_alignment_error,
        "output_alignment_max_abs_error": alignment_error,
        "fixed_joint_max_abs_error": fixed_error,
    }


def link_or_copy(source: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, output)
        return "hardlink"
    except OSError:
        shutil.copy2(source, output)
        return "copy"


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fixed_joints = parse_fixed_joints(args.fixed_joint)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if source.parent != output.parent:
        raise ValueError("Source and output must be sibling datasets on the same mount")

    joint_names, source_info = load_joint_names(source)
    unknown = sorted(set(fixed_joints) - set(joint_names))
    if unknown:
        raise ValueError(f"Unknown fixed joints: {unknown}; available={joint_names}")
    fixed_indices = {joint_names.index(name): value for name, value in fixed_joints.items()}

    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    parquet_reports = []
    linked_files = 0
    copied_files = 0
    skipped_norm_stats = []
    try:
        for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = source_file.relative_to(source)
            output_file = staging / relative
            if relative.parts[0] == "data" and source_file.suffix == ".parquet":
                report = rewrite_parquet(source_file, output_file, fixed_indices)
                report["file"] = str(relative)
                parquet_reports.append(report)
            elif relative.parent == Path(".") and relative.name.startswith("norm_stats"):
                skipped_norm_stats.append(str(relative))
            elif relative == Path("meta/info.json"):
                continue
            else:
                mode = link_or_copy(source_file, output_file)
                linked_files += mode == "hardlink"
                copied_files += mode == "copy"

        output_info = dict(source_info)
        output_info["derived_dataset"] = {
            "source_dataset": str(source),
            "operation": "fix_constant_joints_and_regenerate_future_state_actions",
            "fixed_joint_values": fixed_joints,
            "supervision_source": "corrected observation.state[t+1]",
            "future_shift_frames": 1,
            "last_frame_policy": "repeat_same_episode_last_state",
            "source_action_alignment_verified": True,
            "action_wbc_used": False,
        }
        info_path = staging / "meta" / "info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(
            json.dumps(output_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        summary = {
            "source": str(source),
            "output": str(output),
            "fixed_joint_values": fixed_joints,
            "fixed_joint_indices": {name: joint_names.index(name) for name in fixed_joints},
            "parquet_files": len(parquet_reports),
            "rows": sum(int(item["rows"]) for item in parquet_reports),
            "episodes": sum(int(item["episodes"]) for item in parquet_reports),
            "max_source_alignment_error": max(
                float(item["source_alignment_max_abs_error"]) for item in parquet_reports
            ),
            "max_output_alignment_error": max(
                float(item["output_alignment_max_abs_error"]) for item in parquet_reports
            ),
            "max_fixed_joint_error": max(
                float(item["fixed_joint_max_abs_error"]) for item in parquet_reports
            ),
            "hardlinked_asset_files": int(linked_files),
            "copied_asset_files": int(copied_files),
            "skipped_stale_norm_stats": skipped_norm_stats,
            "parquet_reports": parquet_reports,
        }
        (staging / "fixed_joint_conversion_report.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
