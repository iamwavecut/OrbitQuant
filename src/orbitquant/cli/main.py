from __future__ import annotations

import argparse
import json

from orbitquant import __version__
from orbitquant.eval import list_native_suites
from orbitquant.hub import inspect_model_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitquant")
    parser.add_argument("--version", action="store_true", help="print OrbitQuant version")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect Hugging Face model metadata")
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--revision")
    subparsers.add_parser("native-suites", help="list native eval suites")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "inspect":
        print(json.dumps(inspect_model_metadata(args.model_id, revision=args.revision), indent=2))
        return 0
    if args.command == "native-suites":
        payload = [suite.__dict__ for suite in list_native_suites()]
        print(json.dumps(payload, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
