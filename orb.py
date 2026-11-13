#!/usr/bin/env python3
"""
Opening-range breakout on NSE "stocks in play"  (paper trading and backtest; places NO orders)

Idea (Zarattini, Barbon & Aziz 2024, adapted to NSE cash-segment intraday):
  1. At 09:25, rank Nifty 200 stocks by how unusual their early volume is
     (today's 09:20 candle volume / its average over the previous 14 sessions; the free
     Yahoo feed carries no volume for the 09:15 candle, so the second candle is used).
  2. Take the top few. Trade each only in the direction of its first 10 minutes (09:15-09:25):
     price up over those 10 minutes   -> buy if it later breaks above their high,
     price down over those 10 minutes -> short if it later breaks below their low.
  3. Stop = 25% of the stock's 14-day average daily range away from entry. (The US study used 10%;
     on NSE 5-minute data that is narrower than a single candle, so it cannot be managed or tested.)
  4. No profit target. Exit at the stop or at 15:05, before the broker's auto square-off.
  5. Size every trade so a stop-out costs the same fixed % of capital; intraday margin
     is used only to reach that size, capped at `leverage` times capital in total.

  python orb.py backtest                # compares a fixed set of variants, writes docs/orb_backtest.json
  python orb.py backtest --source demo  # random prices, plumbing check only
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import nifty_scalper as ns

DOCS = Path(__file__).parent / "docs"

# Nifty 200 constituents read from niftyindices.com on 13 Nov 2026. Reviewed every March and September.
NIFTY200 = """
360ONE ABB APLAPOLLO AUBANK ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER ATGL ABCAPITAL AMBUJACEM
APARINDS APOLLOHOSP ASHOKLEY ASIANPAINT ASTRAL AUROPHARMA DMART AXISBANK BSE BAJAJ-AUTO BAJFINANCE BAJAJFINSV
BAJAJHLDNG BANKBARODA BANKINDIA MAHABANK BDL BEL BHARATFORG BHEL BPCL BHARTIARTL GROWW BIOCON BLUESTARCO
BOSCHLTD BRITANNIA CGPOWER CANBK CHOLAFIN CIPLA COALINDIA COCHINSHIP COFORGE COLPAL CONCOR CUMMINSIND DLF
DABUR DIVISLAB DIXON DRREDDY EICHERMOT ETERNAL EXIDEIND NYKAA FEDERALBNK FORTIS GAIL GVT&D GMRAIRPORT
GLENMARK GODFRYPHLP GODREJCP GODREJPROP GRASIM HCLTECH HDFCAMC HDFCBANK HDFCLIFE HAVELLS HEROMOTOCO HINDALCO
HAL HINDCOPPER HINDPETRO HINDUNILVR HINDZINC POWERINDIA HYUNDAI ICICIBANK ICICIGI ICICIAMC IDFCFIRSTB ITC
INDIANB INDHOTEL IOC IRCTC IRFC IREDA INDUSTOWER INDUSINDBK NAUKRI INFY INDIGO JSWENERGY JSWSTEEL JINDALSTEL
JIOFIN JUBLFOOD KEI KALYANKJIL KOTAKBANK LTF LGEINDIA LICHSGFIN LTM LT LAURUSLABS LENSKART LICI LODHA LUPIN
MRF M&MFIN M&M MANKIND MARICO MARUTI MFSL MAXHEALTH MAZDOCK MEESHO MOTILALOFS MPHASIS MCX MUTHOOTFIN NHPC
NLCINDIA NMDC NTPC NATIONALUM NESTLEIND OBEROIRLTY ONGC OIL PAYTM OFSS POLICYBZR PIIND PAGEIND PATANJALI
PERSISTENT PHOENIXLTD PIDILITIND POLYCAB PFC POWERGRID PREMIERENE PRESTIGE PNB RECLTD RADICO RVNL RELIANCE
SBICARD SBILIFE SRF MOTHERSON SHRIRAMFIN ENRIN SIEMENS SOLARINDS SBIN SAIL SUNPHARMA SUPREMEIND SUZLON SWIGGY
TVSMOTOR TATACAP TATACOMM TCS TATACONSUM TMCV TMPV TATAPOWER TATASTEEL TECHM TITAN TORNTPHARM TRENT TIINDIA
UPL ULTRACEMCO UNIONBANK UNITDSPR VBL VAML VEDL VMM IDEA VOLTAS WAAREEENER WIPRO YESBANK ZYDUSLIFE
""".split()


@dataclass
class OrbConfig:
    capital: float = 100_000.0
    risk_pct: float = 0.5          # % of capital lost when one trade is stopped out
    leverage: float = 4.0          # cap on total position value / capital (Groww intraday allows up to 5x)
    top_n: int = 3                 # how many stocks in play to trade each day
    min_relvol: float = 1.0        # first-candle volume must be at least this multiple of normal
    stop_atr: float = 0.25         # stop distance as a fraction of the 14-day average daily range
    lookback: int = 14
    min_price: float = 100.0
    entry_until: str = "14:30"
    exit_at: str = "15:05"
    slippage_bps: float = 3.0      # per market order; breakouts fill worse than calm markets
    fill: str = "conservative"     # how to read a candle that both triggers the entry and touches the stop
    fixed_notional: float = 0.0    # research only: trade this rupee amount per signal, ignore account limits


COST = ns.Config()


def _mins(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def after_range(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[bars.index.hour * 60 + bars.index.minute >= 565]       # candles from 09:25 on


def load_many(symbols, days, source):
    if source == "demo":
        return ns.load_demo(symbols, days, 5)
    if source == "groww":
        return ns.load_groww(symbols, days, 5)
    out = {}
    for i in range(0, len(symbols), 50):               # Yahoo is happier with small batches
        out.update(ns.load_yfinance(symbols[i:i + 50], days, 5))
        if i + 50 < len(symbols):
            time.sleep(1.5)
    return out


def day_table(df: pd.DataFrame, oc: OrbConfig) -> pd.DataFrame:
    """One row per session: first-candle stats, relative volume and ATR, using prior days only."""
    g = df.groupby(df.index.normalize())
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    pc = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(oc.lookback).mean().shift()
    m = df.index.hour * 60 + df.index.minute
    rng = df[(m >= 555) & (m < 565)]                                    # 09:15 and 09:20 candles = 10-minute opening range
    k = rng.groupby(rng.index.normalize())
    first = pd.DataFrame({"or_open": k["open"].first(), "or_high": k["high"].max(), "or_low": k["low"].min(),
                          "or_close": k["close"].last(), "or_n": k["close"].size()})
    vol = df[m == 560]
    first["or_volume"] = vol.set_index(vol.index.normalize())["volume"].replace(0, np.nan)
    d = d.join(first[first["or_n"] == 2].drop(columns="or_n"))
    d["relvol"] = d["or_volume"] / d["or_volume"].rolling(oc.lookback, min_periods=8).mean().shift()
    return d


def picks_for_day(tables: dict, day, oc: OrbConfig) -> list[dict]:
    rows = []
    for sym, t in tables.items():
        if day not in t.index:
            continue
        r = t.loc[day]
        if r[["atr", "relvol", "or_open", "or_high", "or_low", "or_close"]].isna().any():
            continue
        if r["relvol"] < oc.min_relvol or r["or_close"] == r["or_open"] or r["or_open"] < oc.min_price or r["atr"] <= 0:
            continue
        side = 1 if r["or_close"] > r["or_open"] else -1
        rows.append(dict(sym=sym, side=side, level=float(r["or_high"] if side == 1 else r["or_low"]),
                         atr=float(r["atr"]), relvol=float(r["relvol"])))
    return sorted(rows, key=lambda p: -p["relvol"])[: oc.top_n]


def simulate(p: dict, bars: pd.DataFrame, oc: OrbConfig) -> dict | None:
    """Follow one pick through the session. `bars` = that day's candles after the 09:15 one."""
    side, level, sd = p["side"], p["level"], oc.stop_atr * p["atr"]
    slip, t_in, t_out = oc.slippage_bps / 1e4, _mins(oc.entry_until), _mins(oc.exit_at)
    step = pd.Timedelta(minutes=5)
    t = None
    for ts, o, h, l, c in zip(bars.index, bars["open"].to_numpy(), bars["high"].to_numpy(),
                              bars["low"].to_numpy(), bars["close"].to_numpy()):
        m = ts.hour * 60 + ts.minute
        if t is None:
            if m >= t_in:
                return None
            if not (h > level if side == 1 else l < level):
                continue
            base = max(o, level) if side == 1 else min(o, level)          # gap through the level fills at the open
            entry = base * (1 + side * slip)
            stop = entry - side * sd
            t = dict(sym=p["sym"], side=side, sig_time=ts, entry_time=ts, entry=entry, stop=stop, target=None,
                     sd=sd, relvol=p["relvol"], exit=None, exit_time=None, reason="open")
            touched = l <= stop if side == 1 else h >= stop
            beyond = c < stop if side == 1 else c > stop
            if touched and (oc.fill == "conservative" or beyond):         # 5-min data cannot show which came first
                t.update(exit=stop * (1 - side * slip), exit_time=ts + step, reason="stop")
                return t
        else:
            if l <= t["stop"] if side == 1 else h >= t["stop"]:
                px = min(o, t["stop"]) if side == 1 else max(o, t["stop"])
                t.update(exit=px * (1 - side * slip), exit_time=ts + step, reason="stop")
                return t
        if m + 5 >= t_out:
            t.update(exit=c * (1 - side * slip), exit_time=ts + step, reason="time")
            return t
    return t


def size_and_settle(t: dict, equity: float, oc: OrbConfig) -> dict | None:
    if oc.fixed_notional:
        qty = int(oc.fixed_notional // t["entry"])
    else:
        qty = int(min(equity * oc.risk_pct / 100 // t["sd"], equity * oc.leverage / oc.top_n // t["entry"]))
    if qty < 1:
        return None
    t = dict(t, qty=qty, gross=None, cost=None, net=None)
    if t["exit"] is not None:
        buy, sell = (t["entry"], t["exit"]) if t["side"] == 1 else (t["exit"], t["entry"])
        t["gross"] = qty * (sell - buy)
        t["cost"] = ns.order_cost(qty * buy, "B", COST) + ns.order_cost(qty * sell, "S", COST)
        t["net"] = t["gross"] - t["cost"]
    return t


def backtest(data: dict, oc: OrbConfig, universe=None):
    syms = [s for s in data if universe is None or s in universe]
    tables = {s: day_table(data[s], oc) for s in syms}
    days = sorted({d for t in tables.values() for d in t.index})
    by_day = {s: dict(tuple(data[s].groupby(data[s].index.normalize()))) for s in syms}
    equity, trades = oc.capital, []
    for day in days:
        day_net = 0.0
        for p in picks_for_day(tables, day, oc):
            t = simulate(p, after_range(by_day[p["sym"]][day]), oc)
            if t is None or t["exit"] is None:
                continue
            t = size_and_settle(t, equity, oc)
            if t:
                trades.append(t)
                day_net += t["net"]
        if not oc.fixed_notional:
            equity += day_net
    tested = [d for d in days if any(d in t.index and not np.isnan(t.loc[d, "relvol"]) for t in tables.values())]
    return trades, tested


def stats(trades: list[dict], oc: OrbConfig) -> dict:
    if not trades:
        return dict(n=0)
    net = np.array([t["net"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    move = np.array([t["side"] * (t["exit"] - t["entry"]) / t["entry"] * 100 for t in trades])
    r = np.array([t["net"] / (t["qty"] * t["sd"]) for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    eq = np.r_[oc.capital, oc.capital + np.cumsum(net)]
    return dict(n=len(net), win_pct=round(len(wins) / len(net) * 100, 1), avg_move_pct=round(float(move.mean()), 3),
                avg_r=round(float(r.mean()), 3), pf=round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
                gross=round(float(gross.sum())), cost=round(float(gross.sum() - net.sum())), net=round(float(net.sum())),
                net_pct=round(float(net.sum() / oc.capital * 100), 2),
                dd_pct=round(float((np.maximum.accumulate(eq) - eq).max() / oc.capital * 100), 2),
                stops=sum(t["reason"] == "stop" for t in trades))


def row(t: dict) -> dict:
    out = {}
    for c in ["sym", "side", "qty", "sig_time", "entry_time", "entry", "stop", "target", "exit_time", "exit", "reason", "gross", "cost", "net"]:
        v = t.get(c)
        out[c] = None if v is None else str(v) if c.endswith("_time") else v if isinstance(v, (str, int)) else round(float(v), 2)
    return out


VARIANTS = [
    ("A. Base: Nifty 200, top 3 in play, margin up to 4x", {}, "n200"),
    ("B. Same, no margin (1x)", dict(leverage=1.0), "n200"),
    ("C. Top 5 in play", dict(top_n=5), "n200"),
    ("D. Tight stop as in the US study (10% of daily range)", dict(stop_atr=0.10), "n200"),
    ("E. Nifty 50 stocks only", {}, "n50"),
    ("F. Control: every breakout, no in-play filter", dict(top_n=999, min_relvol=0.0, fixed_notional=100_000.0), "n200"),
]


def cmd_backtest(source: str, days: int):
    data = load_many(NIFTY200, days, source)
    if not data:
        sys.exit("No data loaded.")
    base = OrbConfig()
    rows, base_trades, tested = [], [], []
    for name, over, uni in VARIANTS:
        for fill in ("conservative", "optimistic"):
            oc = replace(base, fill=fill, **over)
            trades, tested = backtest(data, oc, set(ns.NIFTY50) if uni == "n50" else None)
            rows.append(dict(name=name, fill=fill, **stats(trades, oc)))
            if not over and fill == "conservative" and uni == "n200":
                base_trades = trades
    out = dict(generated=datetime.now(ns.IST).strftime("%Y-%m-%d %H:%M"), source=source, stocks=len(data),
               days_tested=len(tested), first_day=str(tested[0])[:10] if tested else None,
               last_day=str(tested[-1])[:10] if tested else None, capital=base.capital, risk_pct=base.risk_pct, rows=rows)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "orb_backtest.json").write_text(json.dumps(out, indent=1))
    f = DOCS / "data.json"                                              # show the base case in the dashboard's Backtest tab
    state = json.loads(f.read_text()) if f.exists() else {"status": {}, "trades": {"paper": [], "backtest": []}}
    state["trades"]["backtest"] = [row(t) for t in base_trades]
    state["status"].update(backfill_days=len(tested), capital=base.capital, source=source)
    state["status"].setdefault("now", out["generated"])
    f.write_text(json.dumps(state, indent=1))
    print(f"\nORB BACKTEST  source={source}  stocks={len(data)}  sessions tested={len(tested)}  ({out['first_day']} to {out['last_day']})")
    print(f"{'Variant':<52}{'Fills':<14}{'Trades':>7}{'Win%':>7}{'Move%':>8}{'Avg R':>8}{'PF':>6}{'Gross':>9}{'Costs':>8}{'Net':>9}{'Net%':>8}{'MaxDD%':>8}")
    for r in rows:
        if not r["n"]:
            print(f"{r['name']:<52}{r['fill']:<14}{0:>7}")
            continue
        print(f"{r['name']:<52}{r['fill']:<14}{r['n']:>7}{r['win_pct']:>7}{r['avg_move_pct']:>8}{r['avg_r']:>8}{str(r['pf']):>6}"
              f"{r['gross']:>9,}{r['cost']:>8,}{r['net']:>9,}{r['net_pct']:>8}{r['dd_pct']:>8}")
    print("\nMove% = average price move per trade in the trade's direction, before charges. Avg R = average net result in units of risk.")


def live_scan(state: dict, now: datetime, source: str, capital: float) -> str:
    """One paper-trading pass for today. Mutates state['trades']['paper'] and state['orb_day']; returns an error text or ''."""
    oc = OrbConfig(capital=capital)
    today, mins = now.strftime("%Y-%m-%d"), now.hour * 60 + now.minute
    od = state.get("orb_day") or {}
    if od.get("date") != today:
        if mins < 9 * 60 + 32:
            return ""
        data = load_many(NIFTY200, 32, source)
        if not data:
            return "No market data received"
        day = pd.Timestamp(today)
        tables = {s: day_table(df, oc) for s, df in data.items()}
        if not any(day in t.index and not np.isnan(t.loc[day, "or_volume"]) for t in tables.values()):
            return "Today's opening candles are not available yet (holiday or delayed data)"
        od = dict(date=today, stocks=len(data), picks=picks_for_day(tables, day, oc))
        state["orb_day"] = od
    earlier = [t for t in state["trades"]["paper"] if t["sig_time"][:10] < today]
    equity = capital + sum(t["net"] or 0 for t in earlier)
    todays = []
    if od["picks"]:
        data = load_many([p["sym"] for p in od["picks"]], 2, source)
        cutoff = now - timedelta(minutes=5)                             # ignore the candle still forming
        for p in od["picks"]:
            df = data.get(p["sym"])
            if df is None:
                continue
            bars = after_range(df[(df.index.normalize() == pd.Timestamp(today)) & (df.index <= cutoff)])
            t = simulate(p, bars, oc)
            t = size_and_settle(t, equity, oc) if t else None
            if t:
                todays.append(row(t))
    state["trades"]["paper"] = earlier + todays
    return ""


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "backtest":
        sys.exit(__doc__)
    src = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "yfinance"
    cmd_backtest(src, 59)
