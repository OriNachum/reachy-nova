"""``python -m reachy_nova.harness`` — the supervisor's entry point."""

from .supervisor import main

raise SystemExit(main())
