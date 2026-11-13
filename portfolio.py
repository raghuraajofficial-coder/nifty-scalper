#!/usr/bin/env python3
"""
Paper portfolio for the positional momentum strategy. Places NO orders.

Run once after each market close. It picks up from the last processed day and, for each new
trading day: fills yesterday's queued orders at the open, updates every holding's trailing
stop, reviews the rankings every 20 trading days, and records the day's portfolio value.
State and the dashboard view are both kept in docs/portfolio.json.

Rules: Nifty 200 universe, 12-month momentum skipping the latest month, top 5 in equal amounts,
a holding stays while it ranks in the top 20, and a 15% trailing stop below its highest close.

  python portfolio.py                           # real prices via Yahoo
  python portfolio.py --source demo --replay 90 # random prices, replaying 90 days, to preview the dashboard
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nifty_scalper as ns
import swing

FILE = Path(__file__).parent / "docs" / "portfolio.json"
CFG = swing.Cfg(market_filter=False, trail_pct=0.15)


def new_state() -> dict:
    return dict(started=None, last_day=None, cash=CFG.capital, pos={}, pending=None, stops=[], since_review=-1,
                history=[], closed=[], charges=0.0, fy_gain=0.0, carry=0.0, tax_paid=0.0)


def step(st: dict, d, close, open_, last, mom, nifty):
    pos, day = st["pos"], str(d)[:10]
    if st["pending"] is not None or st["stops"]:                                   # 1. fill queued orders at the open
        target = list(pos) if st["pending"] is None else st["pending"]
        for s in [s for s in pos if s not in target or s in st["stops"]]:
            px = (open_.at[d, s] if np.isfinite(open_.at[d, s]) else last.at[d, s]) * (1 - CFG.slip)
            value = pos[s]["qty"] * px
            fee = swing.delivery_cost(value, "S")
            gain = value - fee - pos[s]["cost"]
            st["cash"] += value - fee
            st["fy_gain"] += gain
            st["charges"] += fee
            st["closed"].append(dict(sym=s, qty=pos[s]["qty"], entry=round(pos[s]["entry"], 2), exit=round(px, 2),
                                     entry_date=pos[s]["date"], exit_date=day, pnl=round(gain, 2),
                                     pnl_pct=round(gain / pos[s]["cost"] * 100, 2),
                                     reason="Trailing stop" if s in st["stops"] else "Fell out of the top 20"))
            del pos[s]
        buys = [s for s in target if s not in pos and s not in st["stops"] and np.isfinite(open_.at[d, s])]
        equity = st["cash"] + sum(p["qty"] * last.at[d, s] for s, p in pos.items())
        for k, s in enumerate(buys):
            px = open_.at[d, s] * (1 + CFG.slip)
            qty = int(min(st["cash"] / (len(buys) - k), equity / CFG.top_n) * 0.995 // px)
            if qty < 1:
                continue
            fee = swing.delivery_cost(qty * px, "B")
            st["cash"] -= qty * px + fee
            st["charges"] += fee
            pos[s] = dict(qty=qty, cost=qty * px + fee, entry=px, date=day, peak=px)
        st["pending"], st["stops"] = None, []
    for s, p in pos.items():                                                       # 2. trailing stops, on the close
        p["peak"] = max(p["peak"], float(last.at[d, s]))
    st["stops"] = [s for s, p in pos.items() if last.at[d, s] < p["peak"] * (1 - CFG.trail_pct)]
    st["since_review"] += 1                                                        # 3. review the rankings every 20 sessions
    if st["since_review"] % CFG.hold == 0:
        rank = mom.loc[d][close.loc[d].notna()].dropna().sort_values(ascending=False).index
        keep = [s for s in pos if s in rank[:CFG.buffer] and s not in st["stops"]]
        st["pending"] = keep + [s for s in rank[:CFG.top_n] if s not in keep][:CFG.top_n - len(keep)]
    if st["last_day"] and st["last_day"][5:7] == "03" and day[5:7] == "04":        # 4. new financial year: settle tax
        taxable = st["fy_gain"] - st["carry"]
        tax = max(0.0, taxable) * swing.TAX
        st["carry"], st["fy_gain"] = max(0.0, -taxable), 0.0
        st["cash"] -= tax
        st["tax_paid"] += tax
    eq = st["cash"] + sum(p["qty"] * last.at[d, s] for s, p in pos.items())        # 5. record the day
    st["history"].append(dict(d=day, eq=round(float(eq), 2), n=round(float(nifty.at[d]), 2)))
    st["started"] = st["started"] or day
    st["last_day"] = day


def view(st: dict, d, close, last, mom, nifty) -> dict:
    pos = st["pos"]
    eq = st["history"][-1]["eq"]
    rank = mom.loc[d][close.loc[d].notna()].dropna().sort_values(ascending=False)
    place = {s: i + 1 for i, s in enumerate(rank.index)}
    holdings = []
    for s, p in pos.items():
        px, stop = float(last.at[d, s]), p["peak"] * (1 - CFG.trail_pct)
        holdings.append(dict(sym=s, qty=p["qty"], entry=round(p["entry"], 2), date=p["date"], last=round(px, 2), peak=round(p["peak"], 2),
                             stop=round(stop, 2), value=round(p["qty"] * px), pnl=round(p["qty"] * px - p["cost"]),
                             pnl_pct=round((p["qty"] * px / p["cost"] - 1) * 100, 1), room_pct=round((px / stop - 1) * 100, 1),
                             rank=place.get(s), selling=s in st["stops"] or (st["pending"] is not None and s not in st["pending"])))
    target = st["pending"] if st["pending"] is not None else list(pos)
    ma = float(nifty.rolling(200).mean().at[d])
    return dict(updated=datetime.now(ns.IST).strftime("%Y-%m-%d %H:%M"), as_of=str(d)[:10], started=st["started"], capital=CFG.capital,
                equity=round(eq), cash=round(st["cash"]), pnl=round(eq - CFG.capital), pnl_pct=round((eq / CFG.capital - 1) * 100, 2),
                charges=round(st["charges"]), tax_paid=round(st["tax_paid"]),
                tax_due=round(max(0.0, st["fy_gain"] - st["carry"]) * swing.TAX), realised=round(st["fy_gain"]),
                next_review=CFG.hold - st["since_review"] % CFG.hold, trail_pct=CFG.trail_pct * 100,
                market=dict(nifty=round(float(nifty.at[d])), ma200=round(ma), above=bool(nifty.at[d] > ma)),
                holdings=sorted(holdings, key=lambda h: -h["value"]),
                buys=[dict(sym=s, last=round(float(last.at[d, s]), 2), rank=place.get(s), mom=round(float(rank[s]) * 100, 1))
                      for s in target if s not in pos and s not in st["stops"]],
                sells=[dict(sym=h["sym"], reason="Trailing stop hit" if h["sym"] in st["stops"] else "Fell out of the top 20")
                       for h in holdings if h["selling"]],
                leaders=[dict(sym=s, mom=round(float(v) * 100, 1), held=s in pos) for s, v in rank.iloc[:10].items()],
                closed=st["closed"][-60:][::-1], history=st["history"][-750:])


def main(source: str, replay: int):
    close, open_, nifty = swing.load(source, "2y")
    close, open_, nifty = close.iloc[-400:], open_.iloc[-400:], nifty.iloc[-400:]
    now = datetime.now(ns.IST)
    if source != "demo" and str(close.index[-1])[:10] == now.strftime("%Y-%m-%d") and now.hour * 60 + now.minute < 15 * 60 + 45:
        close, open_, nifty = close.iloc[:-1], open_.iloc[:-1], nifty.iloc[:-1]      # today's candle is still forming
    last, mom = close.ffill(), close.shift(CFG.skip) / close.shift(CFG.look) - 1
    st = json.loads(FILE.read_text())["state"] if FILE.exists() else new_state()
    todo = close.index[-(replay + 1):] if st["last_day"] is None else close.index[close.index > pd.Timestamp(st["last_day"])]
    for d in todo:
        step(st, d, close, open_, last, mom, nifty)
    if st["last_day"] is None:
        sys.exit("No price data to process.")
    d = pd.Timestamp(st["last_day"])
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(dict(view=view(st, d, close, last, mom, nifty), state=st), indent=1))
    v = json.loads(FILE.read_text())["view"]
    print(f"{v['as_of']}  processed {len(todo)} day(s)  value Rs {v['equity']:,}  holdings {[h['sym'] for h in v['holdings']]}  queued buys {[b['sym'] for b in v['buys']]}")


if __name__ == "__main__":
    a = sys.argv
    main(a[a.index("--source") + 1] if "--source" in a else "yfinance", int(a[a.index("--replay") + 1]) if "--replay" in a else 0)
