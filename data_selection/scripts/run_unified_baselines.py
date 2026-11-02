#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cli import build_parser, execute

if __name__ == "__main__":
    parser = build_parser("unified_offline")
    raise SystemExit(execute(parser.parse_args()))
