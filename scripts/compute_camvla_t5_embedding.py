#!/usr/bin/env python3
"""Cache UMT5-XXL text features from a local Wan Diffusers text encoder.

This is useful when the native Wan ``models_t5_*.pth`` bundle is unavailable.
The output is the plain [tokens, 4096] tensor read by WAMLeRobotDataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, UMT5EncoderModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--wan-text-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    text_root = args.wan_text_root.resolve(strict=True)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if len(tasks) != 1:
        raise ValueError("this experiment requires one unambiguous task prompt")
    instruction = str(tasks.index[0])
    if not instruction:
        raise ValueError("task instruction is empty")
    episode_paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_indices = [
        int(index)
        for path in episode_paths
        for index in pd.read_parquet(path, columns=["episode_index"])["episode_index"]
    ]
    if not episode_indices:
        raise ValueError("dataset has no episodes")
    output = root / "t5_embedding"
    output.mkdir(exist_ok=True)
    if any((output / f"episode_{index:06d}.pt").exists() for index in episode_indices):
        raise FileExistsError("refusing to overwrite an existing T5 embedding")
    tokenizer = AutoTokenizer.from_pretrained(text_root / "tokenizer", local_files_only=True)
    model = UMT5EncoderModel.from_pretrained(
        text_root / "text_encoder", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to(args.device)
    tokenized = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=64)
    tokenized = {key: value.to(args.device) for key, value in tokenized.items()}
    with torch.no_grad():
        embedding = model(**tokenized).last_hidden_state[0].detach().cpu().clone()
    if embedding.ndim != 2 or embedding.shape[1] != 4096 or not torch.isfinite(embedding).all():
        raise ValueError(f"invalid UMT5 embedding shape or values: {tuple(embedding.shape)}")
    for index in episode_indices:
        torch.save(embedding, output / f"episode_{index:06d}.pt")
    print(f"T5 cache: {len(episode_indices)} episodes, {tuple(embedding.shape)}, {instruction}")


if __name__ == "__main__":
    main()
