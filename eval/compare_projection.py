"""Command line over :mod:`react_review.eval_projection`.

    python eval/compare_projection.py OLD.json NEW.json [--key audit_id]

Exit code 1 when anything the OLD artifact recorded has moved; new fields are
printed but never fail the comparison.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from react_review.eval_projection import compare


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compare two eval artifacts on the older schema")
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--key", default="audit_id")
    args = ap.parse_args(argv)

    result = compare(args.old, args.new, key=args.key)
    if result["added_fields"]:
        print(f"new fields ({len(result['added_fields'])}): "
              + ", ".join(result["added_fields"][:12])
              + (" …" if len(result["added_fields"]) > 12 else ""))
    if result["differences"]:
        print(f"PROJECTION DIFFERS on {len(result['differences'])} field(s):")
        for line in result["differences"][:25]:
            print(f"  {line}")
        return 1
    print("projection identical: nothing the old artifact recorded has moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
