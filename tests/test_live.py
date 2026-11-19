#!/usr/bin/env python3
"""
Checks the daily paper-portfolio update against the awkward situations a scheduled run meets:
late index data, half-published days, a run during market hours, a run in the small hours,
holidays, repeat runs, and a price source that has gone backwards. No network. Run: python tests/test_live.py
"""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nifty_scalper as ns
import portfolio
import swing

N = 40                                                     # stocks in the made-up market
DAYS = pd.bdate_range("2025-01-01", "2026-11-18")          # ends on a Wednesday
SYMS = [f"S{i:02d}" for i in range(N)]


def market(upto: str, nifty_upto: str | None = None, thin_last: float = 1.0):
    """Made-up prices up to a date. The index can stop earlier, and the last day can be only partly published."""
    rng = np.random.default_rng(3)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (len(DAYS), N)), axis=0)), index=DAYS, columns=SYMS)
    open_ = close.shift().fillna(close) * 1.001
    nifty = close.mean(axis=1)
    close, open_ = close.loc[:upto].copy(), open_.loc[:upto].copy()
    nifty = nifty.loc[:nifty_upto or upto]
    if thin_last < 1:
        gone = SYMS[int(N * thin_last):]
        close.loc[close.index[-1], gone] = np.nan
        open_.loc[open_.index[-1], gone] = np.nan
    return swing.tidy(close, open_, nifty, None, align=False)[:3]


def at(text: str):
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ns.IST)


def run(upto, now, **kw):
    code = portfolio.main("test", 0, now=at(now), data=market(upto, **kw))
    file = json.loads(portfolio.FILE.read_text()) if portfolio.FILE.exists() else None
    return code, file


def fresh():
    portfolio.FILE = Path(tempfile.mkdtemp()) / "portfolio.json"


def test_normal_evening_run():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-16", "2026-11-16 18:30")
    assert code == 0 and f["view"]["as_of"] == "2026-11-16" and len(f["view"]["holdings"]) == 5


def test_index_data_arrives_late():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-17", "2026-11-18 03:00", nifty_upto="2026-11-16")       # stocks have Tuesday, the index does not
    assert code == 0 and f["view"]["as_of"] == "2026-11-17", "a day must not be dropped because the index is late"
    stale = f["state"]["history"][-1]["n"]
    code, f = run("2026-11-17", "2026-11-18 08:00")                                # the index catches up
    assert f["state"]["history"][-1]["n"] != stale, "the index value should be filled in once it arrives"


def test_half_published_day_waits_then_catches_up():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-17", "2026-11-17 19:00", thin_last=0.5)                 # only half the stocks priced for Tuesday
    assert f["view"]["as_of"] == "2026-11-16" and code == 2, "a half-published day is held back, and the run says it is behind"
    code, f = run("2026-11-17", "2026-11-17 21:00")
    assert code == 0 and f["view"]["as_of"] == "2026-11-17"


def test_run_during_market_hours_ignores_todays_candle():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-17", "2026-11-17 11:00")
    assert code == 0 and f["view"]["as_of"] == "2026-11-16"
    code, f = run("2026-11-17", "2026-11-17 16:00")
    assert code == 0 and f["view"]["as_of"] == "2026-11-17"


def test_small_hours_run_processes_yesterday():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-17", "2026-11-18 03:25")
    assert code == 0 and f["view"]["as_of"] == "2026-11-17"


def test_missed_days_are_caught_up_in_order():
    fresh()
    run("2026-11-12", "2026-11-12 18:30")
    code, f = run("2026-11-17", "2026-11-17 18:30")
    assert [h["d"] for h in f["state"]["history"]][-4:] == ["2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17"]


def test_repeat_run_changes_nothing():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    _, a = run("2026-11-17", "2026-11-17 18:30")
    code, b = run("2026-11-17", "2026-11-17 22:47")
    assert code == 0 and a["state"] == b["state"]


def test_holiday_or_weekend_is_not_a_failure():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    code, f = run("2026-11-13", "2026-11-16 18:30")                                # Monday with no new prices, as on a holiday
    assert code == 0 and f["view"]["as_of"] == "2026-11-13"


def test_stale_for_a_week_is_reported():
    fresh()
    run("2026-11-10", "2026-11-10 18:30")
    code, _ = run("2026-11-10", "2026-11-18 18:30")
    assert code == 2


def test_price_source_behind_the_portfolio_leaves_it_untouched():
    fresh()
    run("2026-11-13", "2026-11-13 18:30")
    _, a = run("2026-11-17", "2026-11-17 18:30")
    before = portfolio.FILE.read_text()
    code, _ = run("2026-11-13", "2026-11-18 03:00")                                # the source has lost the last two days
    assert code == 0 and portfolio.FILE.read_text() == before, "already ahead of the source: leave the file alone and do not report a failure"
    code, _ = run("2026-11-13", "2026-11-27 03:00")                                # ...but still behind a week later is worth reporting
    assert code == 2 and portfolio.FILE.read_text() == before


def test_money_adds_up():
    fresh()
    run("2026-06-01", "2026-06-01 18:30")
    _, f = run("2026-11-17", "2026-11-17 18:30")
    v, st = f["view"], f["state"]
    held = sum(p["qty"] * h["last"] for p, h in ((st["pos"][h["sym"]], h) for h in v["holdings"]))
    assert abs(v["cash"] + held - v["equity"]) < 5, "cash plus holdings must equal the portfolio value"
    assert len(v["holdings"]) <= 5 and v["cash"] >= -1


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   ", name)
        except Exception as e:                             # report every failure, not just the first
            failed += 1
            print("FAIL ", name, "->", repr(e))
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
