import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ace_daily_report import _report_business_state_counts, render_claim_states


def test_entirely_unknown_learning_states_remain_unknown():
    counts = _report_business_state_counts([{"business_metrics": {"states": {
        "accepted": {"yes": 0, "no": 0, "unknown": 4},
        "applied": {"yes": 0, "no": 2, "unknown": 0},
    }}}])
    assert counts["accepted"] is None
    assert counts["applied"] == 0
    assert counts["unknown"]["accepted"] == 4
    assert "| acceptées | inconnu (inconnues=4) |" in render_claim_states({"claim_state_counts": counts})
