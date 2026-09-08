import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import utils


def test_save_state_creates_private_parent_and_replaces_atomically(tmp_path, monkeypatch):
    target = tmp_path / "new-vault" / ".state" / "state.json"
    monkeypatch.setattr(utils, "STATE_FILE", target)
    utils.save_state({"ingested": {"daily.md": {"hash": "before"}}})
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    utils.save_state({"ingested": {"daily.md": {"hash": "after"}}})
    assert json.loads(target.read_text())["ingested"]["daily.md"]["hash"] == "after"
    assert list(target.parent.iterdir()) == [target]


def test_failed_state_write_preserves_previous_file(tmp_path, monkeypatch):
    target = tmp_path / ".state" / "state.json"
    monkeypatch.setattr(utils, "STATE_FILE", target)
    utils.save_state({"ingested": {"existing": True}})
    before = target.read_bytes()
    with pytest.raises(TypeError):
        utils.save_state({"invalid": object()})
    assert target.read_bytes() == before
    assert list(target.parent.iterdir()) == [target]
