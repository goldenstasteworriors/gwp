#!/usr/bin/env python3
"""Run the artifact-bound contact/asymmetric25 training preflight."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.contact_asym25_preflight import preflight_from_environment  # noqa: E402


def main() -> None:
    print(json.dumps(preflight_from_environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
