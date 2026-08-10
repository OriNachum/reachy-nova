"""CLI entry point for the ``reachy-nova-harness`` console script.

Thin argparse dispatcher only — all real behavior lives in
``reachy_nova.harness.supervisor``, imported lazily (inside each subcommand
handler, not at module import time) so that:

- importing this module never pulls in the supervisor's dependencies, and
- running the CLI before the supervisor exists (a sibling task adds it)
  fails with a clean one-line error and exit code 1, never a traceback.
"""

from __future__ import annotations

import argparse
import sys


def _missing_supervisor_message(exc: ImportError) -> str:
    return (
        "reachy-nova-harness: reachy_nova.harness.supervisor is not available yet "
        f"({exc}). This build only ships the CLI stub."
    )


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        from reachy_nova.harness import supervisor
    except ImportError as exc:
        print(_missing_supervisor_message(exc), file=sys.stderr)
        return 1
    return supervisor.run(args)


def _cmd_install_unit(args: argparse.Namespace) -> int:
    try:
        from reachy_nova.harness import supervisor
    except ImportError as exc:
        print(_missing_supervisor_message(exc), file=sys.stderr)
        return 1
    return supervisor.install_unit(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reachy-nova-harness",
        description="Reachy Nova on-device harness supervisor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the harness supervisor in the foreground.")
    run_parser.set_defaults(func=_cmd_run)

    install_parser = subparsers.add_parser(
        "install-unit", help="Install the systemd unit for the harness supervisor."
    )
    install_parser.set_defaults(func=_cmd_install_unit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
