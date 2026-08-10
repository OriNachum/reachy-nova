"""``python -m reachy_nova.harness`` — the supervisor's entry point."""

from .supervisor import main

if __name__ == "__main__":
    raise SystemExit(main())
