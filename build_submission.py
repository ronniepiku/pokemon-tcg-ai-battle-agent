"""Assemble the Kaggle submission bundle.

Stages a clean ``dist/`` directory containing exactly the files the agent needs
at runtime (top-level ``main.py`` and ``deck.csv``, the ``cg`` native package,
the ``bot`` package, and ``model.pth`` if present), then packs it with the
required command:

    tar -czvf submission.tar.gz *

Usage:
    python build_submission.py            # build dist/ and submission.tar.gz
    python build_submission.py --no-tar   # only stage dist/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")

# Files/dirs copied to the top level of the bundle.
TOP_LEVEL_FILES = ["main.py", "deck.csv"]
PACKAGES = ["cg", "bot"]
OPTIONAL_FILES = ["model.pth"]

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

# Required runtime members of the cg package (native libs included).
REQUIRED_CG = ["__init__.py", "api.py", "game.py", "sim.py", "utils.py"]
REQUIRED_CG_LIB = ["cg.dll", "libcg.so"]


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def stage() -> None:
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)

    for fname in TOP_LEVEL_FILES:
        src = os.path.join(ROOT, fname)
        if not os.path.exists(src):
            _fail(f"missing required file: {fname}")
        shutil.copy2(src, os.path.join(DIST, fname))

    for pkg in PACKAGES:
        src = os.path.join(ROOT, pkg)
        if not os.path.isdir(src):
            _fail(f"missing required package: {pkg}/")
        shutil.copytree(src, os.path.join(DIST, pkg), ignore=_IGNORE)

    for fname in OPTIONAL_FILES:
        src = os.path.join(ROOT, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DIST, fname))
            print(f"included optional file: {fname}")
        else:
            print(f"note: {fname} not present (agent runs heuristic-only).")


def verify() -> None:
    # deck.csv must have exactly 60 card IDs.
    with open(os.path.join(DIST, "deck.csv"), encoding="utf-8") as fh:
        cards = [ln for ln in fh.read().splitlines() if ln.strip()]
    if len(cards) != 60:
        _fail(f"deck.csv has {len(cards)} cards; expected 60.")

    cg_dir = os.path.join(DIST, "cg")
    for member in REQUIRED_CG:
        if not os.path.exists(os.path.join(cg_dir, member)):
            _fail(f"cg/{member} missing from bundle.")
    if not any(os.path.exists(os.path.join(cg_dir, lib)) for lib in REQUIRED_CG_LIB):
        _fail("native cg library (cg.dll / libcg.so) missing from bundle.")

    if not os.path.exists(os.path.join(DIST, "main.py")):
        _fail("main.py missing from bundle top level.")
    print("Bundle verification passed.")


def pack() -> None:
    archive = os.path.join(ROOT, "submission.tar.gz")
    if os.path.exists(archive):
        os.remove(archive)
    # Exactly the competition-required command, run from inside dist/.
    subprocess.run(
        ["tar", "-czvf", archive, *sorted(os.listdir(DIST))],
        cwd=DIST,
        check=True,
    )
    size = os.path.getsize(archive) / (1024 * 1024)
    print(f"Created submission.tar.gz ({size:.1f} MB).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the submission bundle.")
    parser.add_argument("--no-tar", action="store_true", help="stage dist/ only")
    args = parser.parse_args()

    stage()
    verify()
    if not args.no_tar:
        pack()
    print("Done. Bundle staged in dist/.")


if __name__ == "__main__":
    main()
