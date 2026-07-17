"""Per-stage sensory logging.

stdlib-only helper that gives every stage of the sensory pipeline (capture,
VAD, event, inject, reaction, ...) one grep-able, parseable INFO-level log
line under the ``nova.sensory`` logger name, so "a sense was heard/handled
correctly" — or silently dropped — is verifiable from the log alone instead
of living only at DEBUG level.

Line shape (fixed, parseable)::

    [SENSE stage=<stage> source=<source> event=<event>] <detail>

Example::

    [SENSE stage=vad source=speech event=3f2a9c1e] utterance detected
"""

from __future__ import annotations

import logging

logger = logging.getLogger("nova.sensory")

_LINE_FORMAT = "[SENSE stage=%s source=%s event=%s] %s"


def stage(stage_name: str, source: str, event: str, detail: str) -> None:
    """Emit one INFO-level, parseable sensory log line for a pipeline stage.

    Args:
        stage_name: the pipeline stage (e.g. ``"vad"``, ``"inject"``,
            ``"capture"``).
        source: the sensory source (e.g. ``"speech"``, ``"vision"``,
            ``"touch"``).
        event: an identifier for this specific sensory event.
        detail: free-form human-readable detail for the line.
    """
    logger.info(_LINE_FORMAT, stage_name, source, event, detail)
