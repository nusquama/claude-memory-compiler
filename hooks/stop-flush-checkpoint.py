"""Claude Stop hook for a throttled, bounded ACE capture.

Claude fires Stop after each assistant response.  This hook performs only
payload validation and filesystem metadata checks.  A successful collector
record or a local launch timestamp keeps the gate at 30 minutes.  When the gate
opens, the collector's source hash handles an unchanged transcript.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from ace_capture import (  # noqa: E402
    cwd_from_payload,
    mark_stop_launch,
    parse_hook_payload,
    spawn_collector,
    stop_is_throttled,
    transcript_from_payload,
)


LOGGER = logging.getLogger("ace.stop")


def main() -> None:
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    payload = parse_hook_payload(sys.stdin.read())
    if payload is None:
        LOGGER.info("SKIP: invalid or missing hook payload")
        return

    transcript_path = transcript_from_payload(payload)
    if transcript_path is None:
        LOGGER.info("SKIP: missing transcript path")
        return

    source_cwd = cwd_from_payload(payload)
    if source_cwd is None:
        LOGGER.info("SKIP: missing source cwd")
        return

    # Do not read transcript tokens on this hot path.  The collector performs
    # the full hash and extraction only when this metadata gate opens.
    if stop_is_throttled(transcript_path):
        return

    if spawn_collector(transcript_path, cwd=source_cwd):
        # This is a launch throttle, not an ACE checkpoint. Durable source
        # state remains owned by the canonical ACE collector.
        mark_stop_launch(transcript_path)
        LOGGER.info("Spawned throttled Claude collector for %s", transcript_path)


if __name__ == "__main__":
    main()
