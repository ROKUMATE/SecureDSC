#!/usr/bin/env python3
"""Download and prepare an English Europarl corpus (one sentence per line).

Pulls the Europarl v7 French-English parallel release (a ~200 MB tarball) and
extracts its clean monolingual English side ``europarl-v7.fr-en.en`` to
``data/europarl-v7.en`` -- the same English corpus used by DeepSC. The training
loader does the lower-casing / punctuation stripping, so this just produces the
raw line-based file.

Usage:
    python -m scripts.get_europarl                 # -> data/europarl-v7.en
    python -m scripts.get_europarl --max-lines 200000 --out data/europarl-v7.en
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request

URL = "https://www.statmt.org/europarl/v7/fr-en.tgz"
MEMBER_SUFFIX = ".fr-en.en"  # the English side inside the tarball


def _download(url: str, dest: str) -> None:
    print(f"Downloading {url} ...", file=sys.stderr)

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(url, dest, _hook)
    print(file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch English Europarl corpus")
    p.add_argument("--out", default="data/europarl-v7.en", help="output text file")
    p.add_argument("--max-lines", type=int, default=None, help="cap number of lines")
    p.add_argument("--url", default=URL, help="override source tarball URL")
    p.add_argument("--keep-tgz", action="store_true", help="keep the downloaded tarball")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = os.path.join(tmp, "europarl.tgz")
        _download(args.url, tgz)
        print("Extracting English side ...", file=sys.stderr)
        with tarfile.open(tgz, "r:gz") as tf:
            member = next(
                (m for m in tf.getmembers() if m.name.endswith(MEMBER_SUFFIX)), None
            )
            if member is None:
                raise RuntimeError(
                    f"No '*{MEMBER_SUFFIX}' file found in {args.url}"
                )
            src = tf.extractfile(member)
            assert src is not None
            n = 0
            with open(args.out, "w", encoding="utf-8") as out:
                for raw in src:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    out.write(line + "\n")
                    n += 1
                    if args.max_lines is not None and n >= args.max_lines:
                        break
        if args.keep_tgz:
            import shutil

            shutil.copy(tgz, os.path.join(os.path.dirname(args.out) or ".", "europarl.tgz"))
    print(f"Wrote {n} lines -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
