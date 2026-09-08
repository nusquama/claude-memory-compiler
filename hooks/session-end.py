"""Claude SessionEnd hook for bounded ACE capture.

The hook parses only routing metadata and launches the canonical ACE collector.
The collector owns transcript parsing, filtering, queue durability, and later
processing.  No model, cursor, summary extractor, or raw transcript copy runs
in this hook.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from ace_capture import (  # noqa: E402
    cwd_from_payload,
    parse_hook_payload,
    spawn_collector,
    transcript_from_payload,
)


LOGGER = logging.getLogger("ace.session-end")


def main() -> None:
    # A collector's Codex child can start Claude-related processes.  Never
    # recurse into this hook from that child.
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

    if spawn_collector(transcript_path, cwd=source_cwd):
        LOGGER.info("Spawned bounded Claude collector for %s", transcript_path)
    else:
        LOGGER.warning("Could not spawn Claude collector for %s", transcript_path)


if __name__ == "__main__":
    main()
