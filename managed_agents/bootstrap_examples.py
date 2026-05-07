#!/usr/bin/env python3
"""
Pre-upload every example IC memo to the Files API and cache file_ids.

`upload_example_memos()` used to re-upload all of `examples/*.pptx` on
every session. This script is the one-time bootstrap that fills
`managed_agents/.examples.json` with `{filename → {file_id, sha256}}`,
so every subsequent session reuses the cached file_ids.

Run this:
  - After cloning the repo and dropping example memos into
    `managed_agents/examples/`, before kicking off your first session.
  - Whenever you add, replace, or remove an example memo.
  - On a fresh deployment to a new Anthropic org (cache is org-scoped).

Usage:
    python -m managed_agents.bootstrap_examples
    python -m managed_agents.bootstrap_examples --no-validate
"""

from __future__ import annotations

import argparse
import sys

from managed_agents.config import EXAMPLES_DIR
from managed_agents.examples_cache import resolve_examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip the per-cached-file get_file roundtrip. Use after "
            "you just uploaded; not for routine maintenance."
        ),
    )
    args = parser.parse_args(argv)

    if not EXAMPLES_DIR.exists():
        print(f"No examples dir at {EXAMPLES_DIR}", file=sys.stderr)
        return 0

    found = sorted(EXAMPLES_DIR.glob("*.pptx"))
    if not found:
        print(f"No .pptx files under {EXAMPLES_DIR}", file=sys.stderr)
        return 0

    print(f"Resolving {len(found)} example memo(s)...")
    resources = resolve_examples(validate_remote=not args.no_validate)

    for r in resources:
        name = r["mount_path"].rsplit("/", 1)[-1]
        print(f"  {name} -> {r['file_id']}")

    print(f"\nCached {len(resources)} examples. "
          f"Future sessions will reference these file_ids directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
