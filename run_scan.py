#!/usr/bin/env python3
"""
One scan pass for free hosting (GitHub Actions). No server needed.

Each run: paper-trades today's market with the scalper rules, refreshes the backtest
once a day, and writes docs/data.json, which the dashboard (docs/index.html) reads.
Places NO real orders.

  python run_scan.py            # does nothing outside 9:20-15:40 IST on weekdays
  python run_scan.py --force    # run regardless of the clock
Settings via environment: SOURCE (yfinance|groww|csv|demo), CAPITAL, BACKFILL_DAYS.
"""
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import nifty_scalper as ns

OUT = Path(__file__).parent / "docs" / "data.json"
SOURCE = os.environ.get("SOURCE", "yfinance")
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "30"))
ARGS = SimpleNamespace(source=SOURCE, symbols_file=os.environ.get("SYMBOLS_FILE"), csv_dir=os.environ.get("CSV_DIR", "data"))
CFG = ns.Config(capital=float(os.environ.get("CAPITAL", "100000")))
COLS = ["sym", "side", "qty", "sig_time", "entry_time", "entry", "stop", "target", "exit_time", "exit", "reason", "gross", "cost", "net"]


def row(t):
    out = {}
    for c in COLS:
        v = t[c]
        out[c] = None if v is None else str(v) if c.endswith("_time") else v if isinstance(v, (str, int)) else round(float(v), 2)
    return out


def main():
    now = datetime.now(ns.IST).replace(tzinfo=None)
    mins, today = now.hour * 60 + now.minute, now.strftime("%Y-%m-%d")
    force = "--force" in sys.argv
    if not force and (now.weekday() >= 5 or not 9 * 60 + 20 <= mins < 15 * 60 + 40):
        print("Outside market hours, nothing to do.")
        return
    state = json.loads(OUT.read_text()) if OUT.exists() else {"status": {}, "trades": {"paper": [], "backtest": []}}
    st, error = state["status"], ""

    # 1. today's paper trades (recomputed from the day's candles so the record stays consistent)
    earlier = [t for t in state["trades"]["paper"] if t["sig_time"][:10] < today]
    cfg = replace(CFG, capital=CFG.capital + sum(t["net"] or 0 for t in earlier))
    try:
        data = ns.load(ARGS, cfg, 5)
    except Exception as e:                                   # noqa: BLE001
        data, error = {}, f"Data download failed: {e}"
    if data:
        cutoff = now - timedelta(minutes=cfg.interval)
        cands = []
        for s, df in data.items():
            cands += [t for t in ns.find_trades(s, df[df.index <= cutoff], cfg) if str(t["sig_time"])[:10] == today]
        taken, _ = ns.run_portfolio(cands, cfg)
        state["trades"]["paper"] = earlier + [row(t) for t in taken]
    elif not error:
        error = "No market data received"

    # 2. backtest tab, refreshed once a day
    if BACKFILL_DAYS > 0 and st.get("backfill_date") != today:
        try:
            hist = ns.load(ARGS, CFG, BACKFILL_DAYS)
            taken, _ = ns.run_portfolio([t for s, df in hist.items() for t in ns.find_trades(s, df, CFG)], CFG)
            if hist:
                state["trades"]["backtest"] = [row(t) for t in taken if t["net"] is not None]
                st["backfill_date"] = today
        except Exception as e:                               # noqa: BLE001
            error = error or f"Backtest failed: {e}"

    st.update(now=now.strftime("%Y-%m-%d %H:%M"), last_scan=now.strftime("%Y-%m-%d %H:%M"), source=SOURCE,
              market_open=now.weekday() < 5 and 9 * 60 + 15 <= mins < 15 * 60 + 30,
              capital=CFG.capital, backfill_days=BACKFILL_DAYS, stocks=len(data), error=error)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(state, indent=1))
    print(f"{st['now']}  stocks={len(data)}  paper trades={len(state['trades']['paper'])}  backtest trades={len(state['trades']['backtest'])}  {error}")


if __name__ == "__main__":
    main()
