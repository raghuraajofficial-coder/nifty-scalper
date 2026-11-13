#!/usr/bin/env python3
"""
One paper-trading pass for free hosting (GitHub Actions). No server needed.

Each run follows today's opening-range breakout picks (see orb.py) and writes
docs/data.json, which the dashboard (docs/index.html) reads. Places NO real orders.

  python run_scan.py            # does nothing outside 9:20-15:40 IST on weekdays
  python run_scan.py --force    # run regardless of the clock
Settings via environment: SOURCE (yfinance|groww|demo), CAPITAL.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nifty_scalper as ns
import orb

OUT = Path(__file__).parent / "docs" / "data.json"
SOURCE = os.environ.get("SOURCE", "yfinance")
CAPITAL = float(os.environ.get("CAPITAL", "100000"))


def main():
    now = datetime.now(ns.IST).replace(tzinfo=None)
    mins = now.hour * 60 + now.minute
    if "--force" not in sys.argv and (now.weekday() >= 5 or not 9 * 60 + 20 <= mins < 15 * 60 + 40):
        print("Outside market hours, nothing to do.")
        return
    state = json.loads(OUT.read_text()) if OUT.exists() else {"status": {}, "trades": {"paper": [], "backtest": []}}
    st = state["status"]
    if st.get("strategy") != "orb":                      # strategy changed: start the paper record afresh
        state["trades"]["paper"] = []
        state.pop("orb_day", None)
    try:
        error = orb.live_scan(state, now, SOURCE, CAPITAL)
    except Exception as e:                               # noqa: BLE001
        error = f"Scan failed: {e}"
    od = state.get("orb_day") or {}
    st.update(strategy="orb", now=now.strftime("%Y-%m-%d %H:%M"), last_scan=now.strftime("%Y-%m-%d %H:%M"), source=SOURCE,
              market_open=now.weekday() < 5 and 9 * 60 + 15 <= mins < 15 * 60 + 30, capital=CAPITAL,
              stocks=od.get("stocks", 0) if od.get("date") == now.strftime("%Y-%m-%d") else 0, error=error,
              picks=[f"{p['sym']} {'long' if p['side'] == 1 else 'short'}" for p in od.get("picks", [])] if od.get("date") == now.strftime("%Y-%m-%d") else [])
    st.setdefault("backfill_days", 0)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(state, indent=1))
    print(f"{st['now']}  picks={st['picks']}  paper trades={len(state['trades']['paper'])}  {error}")


if __name__ == "__main__":
    main()
