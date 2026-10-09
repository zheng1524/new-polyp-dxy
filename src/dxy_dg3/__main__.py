"""Command line entry points for the archived D-G3 package."""
from __future__ import annotations

import argparse
from pathlib import Path

from .archive import verify
from .layout import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="D-G3 frozen archive verification and source package")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="recompute frozen B0 metrics from committed tables")
    check.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if args.command == "verify":
        metrics = verify(args.repo_root.resolve())
        print("D-G3/B0 archive verified")
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
