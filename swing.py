#!/usr/bin/env python3
"""
Short-term positional ("4 to 6 week") momentum on NSE stocks: 10-year backtest. Places NO orders.

Rules (base case)
  1. Universe: today's Nifty 200.
  2. Every 20 trading days, rank stocks by 12-month return, skipping the most recent month.
  3. Hold the top 5 in equal amounts. Keep a holding while it stays in the top 20; replace it otherwise.
  4. Hold cash while the Nifty 50 is below its 200-day average.
  5. Signals use the day's close; orders fill at the next day's open, 0.1% worse than that price.
Costs: Groww delivery charges (brokerage, STT, stamp duty, exchange fees, GST, DP charge on sells)
and 20.8% short-term capital gains tax on each financial year's net realised gain.

Known bias: the universe is TODAY's index members, so stocks that fell out of the index are missing.
That flatters every stock-picking result here. The fair comparison is the equal-weight row, which
holds the same biased universe without any selection.

  python swing.py                 # real prices via Yahoo, writes docs/swing_backtest.json
  python swing.py --source demo   # random prices, plumbing check only
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nifty_scalper as ns
import orb

DOCS = Path(__file__).parent / "docs"
TAX = 0.208                                  # 20% short-term capital gains + 4% cess


@dataclass
class Cfg:
    capital: float = 100_000.0
    top_n: int = 5
    buffer: int = 20                         # keep a holding while it ranks inside this
    look: int = 252                          # momentum window in trading days
    skip: int = 21                           # most recent days ignored (short-term reversal)
    hold: int = 20                           # trading days between reviews
    market_filter: bool = True
    stop_pct: float = 0.0                    # per-stock stop on closing price (0 = off)
    trail_pct: float = 0.0                   # per-stock trailing stop below the highest close since purchase (0 = off)
    slip: float = 0.001


def delivery_cost(value: float, side: str) -> float:
    brok = max(5.0, min(20.0, value * 0.001))
    other = value * (0.0000297 + 0.000001 + 0.000001)
    cost = (brok + other) * 1.18 + value * 0.001                    # GST on fees, then STT
    return cost + (value * 0.00015 if side == "B" else 20 * 1.18)   # stamp duty on buys, DP charge on sells


def load(source: str, period: str = "12y"):
    syms = orb.NIFTY200
    if source == "demo":
        rng = np.random.default_rng(5)
        idx = pd.bdate_range(end=datetime.now().date(), periods=3000)
        drift = rng.normal(0.0004, 0.0004, len(syms))
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, 0.018, (len(idx), len(syms))), axis=0)), index=idx, columns=syms)
        return close, close.shift().fillna(close) * (1 + rng.normal(0, 0.004, close.shape)), close.mean(axis=1)
    import yfinance as yf
    closes, opens = {}, {}
    for i in range(0, len(syms), 50):
        raw = yf.download([f"{s}.NS" for s in syms[i:i + 50]], period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for s in syms[i:i + 50]:
            try:
                df = raw[f"{s}.NS"].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) > 300:
                closes[s], opens[s] = df["Close"], df["Open"]
        time.sleep(1.5)
    nifty = yf.download("^NSEI", period=period, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
    close, open_ = pd.DataFrame(closes), pd.DataFrame(opens)
    for x in (close, open_, nifty):
        x.index = pd.DatetimeIndex(x.index).tz_localize(None)
    close = close[close.index.isin(nifty.index)]
    return close, open_.reindex(close.index), nifty.reindex(close.index).ffill()


def simulate(close, open_, nifty, c: Cfg, start: int):
    dates = close.index
    mom = close.shift(c.skip) / close.shift(c.look) - 1
    ma = nifty.rolling(200).mean()
    last = close.ffill()
    cash, pos, pending, stops = c.capital, {}, None, set()
    fy_gain, carry, tax_paid, charges, n_trades, wins = 0.0, 0.0, 0.0, 0.0, 0, 0
    curve, closed = [], []
    for i in range(start, len(dates)):
        d = dates[i]
        if pending is not None or stops:
            target = list(pos) if pending is None else pending
            for s in [s for s in pos if s not in target or s in stops]:          # sells first
                px = open_.at[d, s] if np.isfinite(open_.at[d, s]) else last.at[d, s]
                value = pos[s]["qty"] * px * (1 - c.slip)
                fee = delivery_cost(value, "S")
                gain = value - fee - pos[s]["cost"]
                cash += value - fee
                fy_gain += gain
                charges += fee
                n_trades += 1
                wins += gain > 0
                closed.append(dict(sym=s, buy=str(dates[pos[s]["day"]])[:10], sell=str(d)[:10], pnl=round(gain),
                                   pct=round(gain / pos[s]["cost"] * 100, 1), weeks=round((i - pos[s]["day"]) / 5, 1),
                                   why="Trailing stop" if s in stops else "Fell out of the top 20"))
                del pos[s]
            buys = [s for s in target if s not in pos and s not in stops and np.isfinite(open_.at[d, s])]
            equity = cash + sum(p["qty"] * last.at[d, s] for s, p in pos.items())
            for k, s in enumerate(buys):
                px = open_.at[d, s] * (1 + c.slip)
                qty = int(min(cash / (len(buys) - k), equity / c.top_n) * 0.995 // px)
                if qty < 1:
                    continue
                fee = delivery_cost(qty * px, "B")
                cash -= qty * px + fee
                charges += fee
                pos[s] = dict(qty=qty, cost=qty * px + fee, entry=px, day=i)
            pending, stops = None, set()
        if c.stop_pct:
            stops = {s for s, p in pos.items() if last.at[d, s] < p["entry"] * (1 - c.stop_pct)}
        if c.trail_pct:
            for s, p in pos.items():
                p["peak"] = max(p.get("peak", p["entry"]), last.at[d, s])
            stops |= {s for s, p in pos.items() if last.at[d, s] < p["peak"] * (1 - c.trail_pct)}
        if (i - start) % c.hold == 0:
            if c.market_filter and not nifty.at[d] > ma.at[d]:
                pending = []
            else:
                rank = mom.loc[d][close.loc[d].notna()].dropna().sort_values(ascending=False).index
                keep = [s for s in pos if s in rank[:c.buffer] and s not in stops]
                pending = keep + [s for s in rank[:c.top_n] if s not in keep][:c.top_n - len(keep)]
        if i + 1 == len(dates) or (d.month == 3 and dates[i + 1].month == 4):   # financial year end: settle tax
            taxable = fy_gain - carry
            tax = max(0.0, taxable) * TAX
            carry = max(0.0, -taxable)
            cash -= tax
            tax_paid += tax
            fy_gain = 0.0
        curve.append((cash + sum(p["qty"] * last.at[d, s] for s, p in pos.items()), tax_paid))
    eq = pd.Series([x[0] for x in curve], index=dates[start:])
    pre = eq + pd.Series([x[1] for x in curve], index=dates[start:])
    g, held = np.array([x["pct"] for x in closed] or [0.0]), np.array([x["weeks"] * 5 for x in closed] or [0])
    return eq, pre, dict(trades=n_trades, win_pct=round(wins / n_trades * 100, 1) if n_trades else 0,
                         charges=round(charges), tax=round(tax_paid),
                         trade_avg_pct=round(float(g.mean()), 1), trade_median_pct=round(float(np.median(g)), 1),
                         avg_win_pct=round(float(g[g > 0].mean()), 1) if (g > 0).any() else 0.0,
                         avg_loss_pct=round(float(g[g <= 0].mean()), 1) if (g <= 0).any() else 0.0,
                         best_trade_pct=round(float(g.max()), 1), worst_trade_pct=round(float(g.min()), 1),
                         avg_hold_weeks=round(float(held.mean()) / 5, 1), median_hold_weeks=round(float(np.median(held)) / 5, 1)), closed


def perf(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    yr = eq.groupby(eq.index.year).last().pct_change()
    yr.iloc[0] = eq.groupby(eq.index.year).last().iloc[0] / eq.iloc[0] - 1
    daily = eq.pct_change().dropna()
    mo = eq.groupby([eq.index.year, eq.index.month]).last().pct_change().dropna()
    monthly = dict(avg=round(float(mo.mean() * 100), 2), median=round(float(mo.median() * 100), 2),
                   up_pct=round(float((mo > 0).mean() * 100), 1), best=round(float(mo.max() * 100), 1), worst=round(float(mo.min() * 100), 1),
                   months=len(mo), above_5=int((mo > 0.05).sum()), below_minus_5=int((mo < -0.05).sum()),
                   worst_run=int(max((len(list(g)) for k, g in __import__("itertools").groupby(mo < 0) if k), default=0)),
                   last_24={f"{y}-{m:02d}": round(float(v) * 100, 1) for (y, m), v in mo.iloc[-24:].items()})
    return dict(cagr=round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100, 1), final=round(float(eq.iloc[-1])),
                maxdd=round(float((1 - eq / eq.cummax()).max() * 100), 1), vol=round(float(daily.std() * np.sqrt(252) * 100), 1),
                worst_year=round(float(yr.min() * 100), 1), best_year=round(float(yr.max() * 100), 1),
                down_years=int((yr < 0).sum()), monthly=monthly, years={int(k): round(float(v) * 100, 1) for k, v in yr.items()})


FEATURED = "J."                               # the version the paper portfolio follows
VARIANTS = [
    ("A. Base: 12-month momentum, top 5, review every 4 weeks, market filter", {}),
    ("B. Base without the market filter", dict(market_filter=False)),
    ("C. Base on 6-month momentum", dict(look=126)),
    ("D. Base holding the top 10", dict(top_n=10)),
    ("E. Base with a 10% stop-loss per stock", dict(stop_pct=0.10)),
    ("F. Base reviewed every 6 weeks", dict(hold=30)),
    ("G. Base with a 10% trailing stop per stock", dict(trail_pct=0.10)),
    ("H. Base with a 15% trailing stop per stock", dict(trail_pct=0.15)),
    ("I. Base with a 20% trailing stop per stock", dict(trail_pct=0.20)),
    ("J. No market filter, 15% trailing stop per stock", dict(market_filter=False, trail_pct=0.15)),
]


def main(source: str):
    close, open_, nifty = load(source)
    start = Cfg().look + 2
    rows, feat = [], None
    for name, over in VARIANTS:
        eq, pre, extra, log = simulate(close, open_, nifty, replace(Cfg(), **over), start)
        rows.append(dict(name=name, after_tax=perf(eq), before_tax_cagr=perf(pre)["cagr"], **extra))
        if name.startswith(FEATURED):
            feat, feat_log = eq, log
    span = close.index[start:]
    bench = {"Nifty 50 index, buy and hold (dividends excluded)": nifty.loc[span] / nifty.loc[span[0]] * 100_000,
             "Equal weight of the same 200 stocks, no selection": (1 + close.pct_change(fill_method=None).loc[span].mean(axis=1).fillna(0)).cumprod() * 100_000}
    out = dict(generated=datetime.now(ns.IST).strftime("%Y-%m-%d %H:%M"), source=source, stocks=close.shape[1],
               first_day=str(span[0])[:10], last_day=str(span[-1])[:10], capital=100_000, rows=rows,
               benchmarks=[dict(name=k, **perf(v)) for k, v in bench.items()])
    if feat is not None:                                   # month-end series for the dashboard's chart and heatmap
        key = [feat.index.year, feat.index.month]
        me = pd.DataFrame({"s": feat, **{k: v for k, v in zip("ne", bench.values())}}).groupby(key).last()
        mo = feat.groupby(key).last().pct_change()
        out["featured"] = dict(name=next(r["name"] for r in rows if r["name"].startswith(FEATURED)),
                               curve=[[f"{y}-{m:02d}", round(float(r.s)), round(float(r.n)), round(float(r.e))] for (y, m), r in me.iterrows()],
                               months={f"{y}-{m:02d}": round(float(v) * 100, 1) for (y, m), v in mo.dropna().items()},
                               rupees={f"{y}-{m:02d}": round(float(v)) for (y, m), v in feat.groupby(key).last().diff().dropna().items()},
                               trades=feat_log)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "swing_backtest.json").write_text(json.dumps(out, indent=1))
    print(f"\nSWING BACKTEST  source={source}  stocks={close.shape[1]}  {out['first_day']} to {out['last_day']}")
    print(f"{'Variant':<74}{'CAGR%':>7}{'preTax':>8}{'MaxDD%':>8}{'Worst':>7}{'Best':>7}{'Trades':>8}{'Win%':>6}{'Charges':>9}{'Tax':>9}{'Final':>11}")
    for r in rows:
        a = r["after_tax"]
        print(f"{r['name']:<74}{a['cagr']:>7}{r['before_tax_cagr']:>8}{a['maxdd']:>8}{a['worst_year']:>7}{a['best_year']:>7}"
              f"{r['trades']:>8}{r['win_pct']:>6}{r['charges']:>9,}{r['tax']:>9,}{a['final']:>11,}")
    for b in out["benchmarks"]:
        print(f"{b['name']:<74}{b['cagr']:>7}{'':>8}{b['maxdd']:>8}{b['worst_year']:>7}{b['best_year']:>7}{'':>32}{b['final']:>11,}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "yfinance")
