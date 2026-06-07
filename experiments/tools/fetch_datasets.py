"""One-time dataset fetcher — the ONLY network-touching code in the harness.

For each registered task it downloads the pinned raw source file from its URL,
verifies the bytes against the SHA256 recorded in ``prepare._TASK_REGISTRY``, and
writes it under ``experiments/data/<task>/``. After this, ``prepare.load_task``
reads exclusively from ``data/`` and never opens a socket.

This module is **never imported by prepare.py** (that is the offline guarantee).
It imports ``prepare`` only to read the registry and ``DATA_DIR``.

Usage:
    uv run python tools/fetch_datasets.py                 # fetch + verify all tasks
    uv run python tools/fetch_datasets.py --task adult    # one task
    uv run python tools/fetch_datasets.py --verify-only   # no network; checksum
                                                          #   the committed files

In ``--verify-only`` mode nothing is downloaded: it recomputes the SHA256 of the
already-committed files and compares them to the registry. Use it in CI / as a
provenance check. A registry whose checksum is still a ``__SHA_*__`` placeholder
is reported (so a first population can paste in the printed digest), but is never
silently treated as a pass.
"""

import argparse
import hashlib
import os
import sys
import urllib.request

# Put the experiments/ root on sys.path so `import prepare` resolves when this is
# run as `python tools/fetch_datasets.py` (tools/ would otherwise be sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_real_sha(value) -> bool:
    """True only for a concrete 64-char hex digest (not a placeholder)."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "autoresearch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_task(task: str, *, verify_only: bool) -> list:
    """Fetch (or verify) every committed file for ``task``.

    Returns a list of ``(filename, status, actual_sha)`` tuples. ``status`` is one
    of: ``OK`` (bytes match the registry), ``MISMATCH`` (bytes differ — a hard
    error), ``MISSING`` (verify-only, file absent), ``RECORDED`` (downloaded but
    the registry sha is still a placeholder — paste ``actual_sha`` in).
    """
    entry = prepare._TASK_REGISTRY[task]
    files = entry.get("files")
    if not files:
        raise SystemExit(f"Task {task!r} has no `files` entry in the registry.")

    task_dir = os.path.join(prepare.DATA_DIR, task)
    if not verify_only:
        os.makedirs(task_dir, exist_ok=True)

    results = []
    for filename, expected in files.items():
        dest = os.path.join(task_dir, filename)

        if verify_only:
            if not os.path.exists(dest):
                results.append((filename, "MISSING", None))
                continue
            with open(dest, "rb") as fh:
                actual = _sha256(fh.read())
            if not _is_real_sha(expected):
                results.append((filename, "NO_EXPECTED_SHA", actual))
            elif actual == expected:
                results.append((filename, "OK", actual))
            else:
                results.append((filename, "MISMATCH", actual))
            continue

        # Download mode (network).
        data = _download(entry["url"])
        actual = _sha256(data)
        if not _is_real_sha(expected):
            status = "RECORDED"
        elif actual == expected:
            status = "OK"
        else:
            status = "MISMATCH"
        with open(dest, "wb") as fh:
            fh.write(data)
        results.append((filename, status, actual))

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=sorted(prepare._TASK_REGISTRY),
        help="fetch/verify a single task (default: all registered tasks)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="no network; checksum the already-committed files against the registry",
    )
    args = parser.parse_args(argv)

    tasks = [args.task] if args.task else sorted(prepare._TASK_REGISTRY)
    bad = False
    for task in tasks:
        results = fetch_task(task, verify_only=args.verify_only)
        for filename, status, actual in results:
            flag = "" if status in ("OK", "RECORDED", "NO_EXPECTED_SHA") else "  <-- FAIL"
            sha = actual if actual is not None else "(absent)"
            print(f"{task:<16} {filename:<20} {status:<16} {sha}{flag}")
            if status in ("MISMATCH", "MISSING"):
                bad = True

    if bad:
        print("\nFAIL: one or more files mismatched or are missing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
