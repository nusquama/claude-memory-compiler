from __future__ import annotations

import os
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import scan_md  # noqa: E402


def test_explicit_scan_source_wins_over_ace_destination() -> None:
    source = Path("/tmp/ace-source-checkout")
    destination = Path("/tmp/ace-destination-vault")
    observed: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return CompletedProcess(command, 0, stdout=f"{source}\n", stderr="")

    with patch.dict(
        os.environ,
        {
            "ACE_PROJECT_DIR": str(destination),
            "CLAUDE_PROJECT_DIR": str(destination),
            "PWD": str(destination),
        },
    ), patch.object(scan_md.subprocess, "run", side_effect=fake_run):
        assert scan_md.find_repo_root(None, str(source)) == source.resolve()

    assert observed["command"][:3] == ["git", "-C", str(source)]
