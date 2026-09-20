from __future__ import annotations

from jevguard_nsfa.compare import render_markdown


def test_compare_report_keeps_missing_cost_as_na() -> None:
    jev = {
        "model": "jev-test",
        "mode": "managed-api-online",
        "quality": {"binary": {"f1": 0.9}},
        "usage": {"cost_per_1000_successful_requests_usd": 0.01},
    }
    sing = {
        "model": "sing-test",
        "mode": "local-realtime-classification",
        "quality": {"binary": {"f1": 0.8}},
        "usage": {"cost_per_1000_successful_requests_usd": None},
    }
    report = render_markdown(jev, sing)
    assert "0.9000" in report
    assert "0.8000" in report
    assert "n/a" in report
