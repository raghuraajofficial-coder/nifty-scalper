#!/usr/bin/env python3
"""
Nifty 50 intraday signal scanner + backtester  (v0: SIGNALS ONLY - places no orders)

Commands
  costs      show what a round trip costs at different position sizes
  backtest   run the rules over past 5-minute candles, net of all charges
  scan       watch the market live and print BUY / SELL / EXIT signals

Data sources (--source)
  yfinance   free, no login, up to ~59 days of 5-min candles (can lag a few minutes)
  groww      Groww Trade API (needs subscription; token read from env GROWW_API_TOKEN)
  csv        a folder of SYMBOL.csv files: datetime,open,high,low,close,volume (IST)
  demo       random synthetic prices - only to check the tool runs; results mean nothing
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

# Nifty 50 as of 30 Sep 2026 (BSE replaced WIPRO). Reviewed every March and September:
# check against niftyindices.com, or pass your own list with --symbols-file.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "BSE", "CIPLA", "COALINDIA",
    "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TMPV", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
]


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    # money
    capital: float = 100_000.0
    leverage: float = 1.0          # 1 = no intraday margin. Groww allows up to 5x; keep 1 until proven.
    risk_pct: float = 0.5          # % of capital lost if the stop is hit
    # daily guard rails
    max_trades_day: int = 3
    max_losses_day: int = 2
    daily_loss_pct: float = 1.5    # stop for the day after losing this much
    # strategy
    interval: int = 5              # candle size in minutes
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_lo: float = 50.0
    rsi_hi: float = 70.0
    vol_mult: float = 1.2          # signal candle volume vs 20-candle average
    atr_stop: float = 1.0          # stop distance in ATRs
    rr: float = 1.5                # target = rr x stop distance
    min_edge_mult: float = 3.0     # skip unless target profit >= this x round-trip charges
    long_only: bool = False
    entry_start: str = "09:30"
    entry_end: str = "14:45"
    square_off: str = "15:10"      # Groww auto square-off is around this time
    slippage_bps: float = 2.0      # per market order
    # charges: NSE equity intraday on Groww (check groww.in/pricing before trusting)
    brok_flat: float = 20.0
    brok_pct: float = 0.001
    brok_min: float = 5.0
    stt_sell_pct: float = 0.00025
    exch_pct: float = 0.0000297
    sebi_pct: float = 0.000001
    ipft_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18


def order_cost(value: float, side: str, c: Config) -> float:
    """All charges on one executed order. side = 'B' or 'S'."""
    brokerage = max(c.brok_min, min(c.brok_flat, value * c.brok_pct))
    exch, sebi, ipft = value * c.exch_pct, value * c.sebi_pct, value * c.ipft_pct
    gst = c.gst_pct * (brokerage + exch + sebi + ipft)
    stt = value * c.stt_sell_pct if side == "S" else 0.0
    stamp = value * c.stamp_buy_pct if side == "B" else 0.0
    return brokerage + exch + sebi + ipft + gst + stt + stamp


def round_trip_cost(value: float, c: Config) -> float:
    return order_cost(value, "B", c) + order_cost(value, "S", c)


# ----------------------------------------------------------------------------- data
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    t = df.index.time
    return df[(t >= datetime.strptime("09:15", "%H:%M").time()) & (t < datetime.strptime("15:30", "%H:%M").time())]


def load_yfinance(symbols, days, interval):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run:  pip install yfinance")
    tick = {s: f"{s}.NS" for s in symbols}
    raw = yf.download(list(tick.values()), period=f"{min(days, 59)}d", interval=f"{interval}m",
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    for s, t in tick.items():
        try:
            df = raw[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if df.empty:
            continue
        df = df.rename(columns=str.lower)
        idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
        df.index = idx.tz_convert(IST).tz_localize(None)
        out[s] = _clean(df)
    return out


def load_groww(symbols, days, interval):
    try:
        from growwapi import GrowwAPI
    except ImportError:
        sys.exit("growwapi is not installed. Run:  pip install growwapi")
    token = os.environ.get("GROWW_API_TOKEN")
    if not token:
        sys.exit("Set the GROWW_API_TOKEN environment variable first (never paste it into code or chat).")
    g = GrowwAPI(token)
    const = getattr(g, f"CANDLE_INTERVAL_MIN_{interval}")
    end = datetime.now(IST).replace(tzinfo=None)
    start = end - timedelta(days=days)
    fmt = "%Y-%m-%d %H:%M:%S"
    out = {}
    for s in symbols:
        rows, a = [], start
        while a < end:                                  # API allows 30 days per call for 5-min candles
            b = min(a + timedelta(days=30), end)
            try:
                resp = g.get_historical_candles(exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                                                groww_symbol=f"NSE-{s}", start_time=a.strftime(fmt),
                                                end_time=b.strftime(fmt), candle_interval=const)
                rows += resp.get("candles", [])
            except Exception as e:                      # noqa: BLE001
                print(f"  ! {s}: {e}")
            a = b
            time.sleep(0.25)
        if not rows:
            continue
        df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
        if pd.api.types.is_numeric_dtype(df["ts"]):     # epoch seconds
            df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
        else:                                           # docs give 'yyyy-MM-ddTHH:mm:ss', assumed IST
            df.index = pd.to_datetime(df["ts"])
        out[s] = _clean(df)
    return out


def load_csv(symbols, folder):
    out = {}
    for s in symbols:
        p = Path(folder) / f"{s}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [x.lower() for x in df.columns]
        df.index = pd.to_datetime(df["datetime"])
        out[s] = _clean(df)
    return out


def load_demo(symbols, days, interval):
    """Random walk prices. For checking the plumbing only."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=datetime.now(IST).date(), periods=max(days * 5 // 7, 3))
    bars = (375 // interval)
    out = {}
    for s in symbols:
        price = float(rng.uniform(300, 4000))
        idx, rows = [], []
        for d in dates:
            drift = rng.normal(0, 0.0003)
            for k in range(bars):
                u = 1.6 - 1.2 * np.sin(np.pi * k / (bars - 1))           # busier at open and close
                o = price
                c_ = o * (1 + drift + rng.normal(0, 0.0011) * u)
                h = max(o, c_) * (1 + abs(rng.normal(0, 0.0004)) * u)
                l = min(o, c_) * (1 - abs(rng.normal(0, 0.0004)) * u)
                v = rng.lognormal(10, 0.6) * u
                idx.append(d + timedelta(hours=9, minutes=15 + k * interval))
                rows.append((o, h, l, c_, v))
                price = c_
            price *= 1 + rng.normal(0, 0.004)                           # overnight gap
        out[s] = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close", "volume"])
    return out


def load(args, c: Config, days: int):
    syms = NIFTY50
    if args.symbols_file:
        syms = [x.strip().upper() for x in Path(args.symbols_file).read_text().split() if x.strip()]
    if args.source == "yfinance":
        return load_yfinance(syms, days, c.interval)
    if args.source == "groww":
        return load_groww(syms, days, c.interval)
    if args.source == "csv":
        return load_csv(syms, args.csv_dir)
    return load_demo(syms, days, c.interval)


# ----------------------------------------------------------------------------- strategy
def add_indicators(df: pd.DataFrame, c: Config) -> pd.DataFrame:
    df = df.copy()
    day = df.index.normalize()
    tp = (df.high + df.low + df.close) / 3
    df["vwap"] = (tp * df.volume).groupby(day).cumsum() / df.volume.groupby(day).cumsum().replace(0, np.nan)
    df["ema_f"] = df.close.ewm(span=c.ema_fast, adjust=False).mean()
    df["ema_s"] = df.close.ewm(span=c.ema_slow, adjust=False).mean()
    d = df.close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_avg"] = df.volume.rolling(20).mean().shift()
    return df


def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def find_trades(sym: str, df: pd.DataFrame, c: Config) -> list[dict]:
    """Every setup in one stock, with how it played out (qty-independent).

    Trend-pullback rules, long side (short is the mirror image):
      1. price above the day's VWAP and EMA9 above EMA21        -> intraday uptrend
      2. this or the previous candle dipped to the EMA9         -> a pullback, not a chase
      3. candle closes green and above the previous high        -> buyers are back
      4. RSI between 50 and 70, volume above 1.2x average       -> momentum without exhaustion
    Entry at the next candle's open, stop 1 ATR away, target 1.5x the stop, flat by 15:10.
    """
    if len(df) < 40:
        return []
    df = add_indicators(df, c)
    o, h, l, cl, v = (df[x].to_numpy() for x in ["open", "high", "low", "close", "volume"])
    vw, ef, es, rsi, atr, va = (df[x].to_numpy() for x in ["vwap", "ema_f", "ema_s", "rsi", "atr", "vol_avg"])
    idx = df.index
    day = idx.normalize().to_numpy()
    mins = (idx.hour * 60 + idx.minute).to_numpy()
    t0, t1, tsq = _hm(c.entry_start), _hm(c.entry_end), _hm(c.square_off)
    step = pd.Timedelta(minutes=c.interval)
    slip = c.slippage_bps / 10_000
    n, trades, i = len(df), [], 25
    while i < n:
        ok = (day[i] == day[i - 1] and t0 <= mins[i] + c.interval <= t1
              and not np.isnan([vw[i], rsi[i], atr[i], va[i]]).any() and atr[i] > 0 and va[i] > 0
              and v[i] >= c.vol_mult * va[i])
        side = 0
        if ok:
            if (cl[i] > vw[i] and ef[i] > es[i] and min(l[i], l[i - 1]) <= ef[i] * 1.0005
                    and cl[i] > o[i] and cl[i] > h[i - 1] and c.rsi_lo <= rsi[i] <= c.rsi_hi):
                side = 1
            elif (not c.long_only and cl[i] < vw[i] and ef[i] < es[i] and max(h[i], h[i - 1]) >= ef[i] * 0.9995
                  and cl[i] < o[i] and cl[i] < l[i - 1] and 100 - c.rsi_hi <= rsi[i] <= 100 - c.rsi_lo):
                side = -1
        if side == 0:
            i += 1
            continue
        sd = c.atr_stop * atr[i]
        t = dict(sym=sym, side=side, sig_time=idx[i] + step, relvol=v[i] / va[i], sd=sd,
                 entry=cl[i], stop=cl[i] - side * sd, target=cl[i] + side * c.rr * sd,
                 entry_time=None, exit=None, exit_time=None, reason="open")
        j = i + 1
        if j >= n:                       # live: signal candle just closed, no fill candle yet
            trades.append(t)
            break
        if day[j] != day[i]:
            i += 1
            continue
        entry = o[j] * (1 + side * slip)
        stop, target = entry - side * sd, entry + side * c.rr * sd
        t.update(entry=entry, stop=stop, target=target, entry_time=idx[j])
        k = j
        while k < n and day[k] == day[j]:
            hit_stop = l[k] <= stop if side == 1 else h[k] >= stop
            hit_tgt = h[k] >= target if side == 1 else l[k] <= target
            if hit_stop:                 # if stop and target fall in the same candle, assume the stop
                px = min(o[k], stop) if side == 1 else max(o[k], stop)
                t.update(exit=px * (1 - side * slip), reason="stop")
            elif hit_tgt:
                t.update(exit=target, reason="target")
            elif mins[k] + c.interval >= tsq or k + 1 >= n or day[k + 1] != day[j]:
                if not (k + 1 >= n and mins[k] + c.interval < tsq):     # data simply ended -> still open
                    t.update(exit=cl[k] * (1 - side * slip), reason="time")
            if t["exit"] is not None:
                t["exit_time"] = idx[k] + step
                break
            k += 1
        trades.append(t)
        i = k + 1
    return trades


def run_portfolio(cands: list[dict], c: Config):
    """Apply the account rules: one position at a time, sizing, cost filter, daily limits."""
    cands = sorted(cands, key=lambda t: (t["sig_time"], -t["relvol"]))
    equity, busy_until, taken = c.capital, pd.Timestamp.min, []
    cur_day, d_trades, d_losses, d_pnl, d_start = None, 0, 0, 0.0, equity
    for t in cands:
        d = t["sig_time"].date()
        if d != cur_day:
            cur_day, d_trades, d_losses, d_pnl, d_start = d, 0, 0, 0.0, equity
        if t["sig_time"] < busy_until:
            continue
        if d_trades >= c.max_trades_day or d_losses >= c.max_losses_day or d_pnl <= -c.daily_loss_pct / 100 * d_start:
            continue
        qty = int(min(equity * c.risk_pct / 100 // t["sd"], equity * c.leverage // t["entry"]))
        if qty < 1:
            continue
        value = qty * t["entry"]
        est_cost = round_trip_cost(value, c)
        if qty * c.rr * t["sd"] < c.min_edge_mult * est_cost:      # move too small to be worth the charges
            continue
        t = dict(t, qty=qty, value=value, est_cost=est_cost, gross=None, cost=None, net=None)
        d_trades += 1
        if t["exit"] is None:
            busy_until = pd.Timestamp.max
        else:
            buy, sell = (t["entry"], t["exit"]) if t["side"] == 1 else (t["exit"], t["entry"])
            t["gross"] = qty * (sell - buy)
            t["cost"] = order_cost(qty * buy, "B", c) + order_cost(qty * sell, "S", c)
            t["net"] = t["gross"] - t["cost"]
            equity += t["net"]
            d_pnl += t["net"]
            d_losses += t["net"] < 0
            busy_until = t["exit_time"]
        taken.append(t)
    return taken, equity


# ----------------------------------------------------------------------------- commands
def cmd_costs(args, c: Config):
    print("\nRound-trip charges, NSE equity intraday on Groww (buy + sell)\n")
    print(f"{'Position':>12} {'Charges':>10} {'Break-even move':>16}")
    for v in [10_000, 25_000, 50_000, 100_000, 200_000, 500_000]:
        rt = round_trip_cost(v, c)
        print(f"{v:>12,.0f} {rt:>10,.2f} {rt / v * 100:>15.3f}%")
    print("\nA scalp has to move more than the break-even % before it earns a rupee.\n")


def cmd_backtest(args, c: Config):
    data = load(args, c, args.days)
    if not data:
        sys.exit("No data loaded.")
    days = sorted({d for df in data.values() for d in np.unique(df.index.date)})
    cands = [t for s, df in data.items() for t in find_trades(s, df, c)]
    taken, equity = run_portfolio(cands, c)
    done = [t for t in taken if t["net"] is not None]
    print(f"\nBACKTEST  source={args.source}  stocks={len(data)}  days={len(days)}  ({days[0]} to {days[-1]})")
    if args.source == "demo":
        print("*** DEMO DATA: random prices. These numbers say nothing about real markets. ***")
    print(f"Setups found: {len(cands)}   trades taken after account rules: {len(done)}")
    if not done:
        print("No trades. Loosen the filters or load more days.\n")
        return
    net = np.array([t["net"] for t in done])
    gross, cost = sum(t["gross"] for t in done), sum(t["cost"] for t in done)
    wins, losses = net[net > 0], net[net <= 0]
    eq = c.capital + np.cumsum(net)
    dd = (np.maximum.accumulate(np.r_[c.capital, eq]) - np.r_[c.capital, eq]).max()
    daily = pd.Series(net, index=[t["sig_time"].date() for t in done]).groupby(level=0).sum().reindex(days, fill_value=0.0)
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    rows = [
        ("Win rate", f"{len(wins) / len(net) * 100:.1f}%"),
        ("Average win / loss", f"Rs {wins.mean() if len(wins) else 0:,.0f} / Rs {losses.mean() if len(losses) else 0:,.0f}"),
        ("Profit factor", f"{pf:.2f}   (above 1.0 = profitable; want 1.3+ before trusting it)"),
        ("Gross P&L", f"Rs {gross:,.0f}"),
        ("Charges paid", f"Rs {cost:,.0f}"),
        ("NET P&L", f"Rs {net.sum():,.0f}   ({net.sum() / c.capital * 100:+.2f}% on Rs {c.capital:,.0f})"),
        ("Worst drawdown", f"Rs {dd:,.0f}   ({dd / c.capital * 100:.1f}%)"),
        ("Average per day", f"{daily.mean() / c.capital * 100:+.3f}%   (your target: +1 to +2%)"),
        ("Green / red / flat days", f"{(daily > 0).sum()} / {(daily < 0).sum()} / {(daily == 0).sum()}"),
        ("Exits", ", ".join(f"{k}: {n}" for k, n in pd.Series([t['reason'] for t in done]).value_counts().items())),
    ]
    for k, val in rows:
        print(f"  {k:<26}{val}")
    if len(done) < 100:
        print(f"\n  Only {len(done)} trades: too few to judge. Treat any result here as noise.")
    if args.out:
        cols = ["sym", "side", "sig_time", "entry_time", "entry", "stop", "target", "qty", "exit_time", "exit", "reason", "gross", "cost", "net"]
        pd.DataFrame(done)[cols].to_csv(args.out, index=False)
        print(f"\n  Trade list saved to {args.out}")
    print()


def _fmt_signal(t, kind):
    if kind == "ENTRY":
        act = "BUY " if t["side"] == 1 else "SELL (short)"
        risk = t["qty"] * t["sd"]
        return (f"{act} {t['sym']}  qty {t['qty']}  ~Rs {t['entry']:.2f}  | stop {t['stop']:.2f}  target {t['target']:.2f}"
                f"  | risk Rs {risk:,.0f}  charges ~Rs {t['est_cost']:.0f}")
    act = "SELL to close" if t["side"] == 1 else "BUY to close"
    return f"EXIT {t['sym']}  {act} ~Rs {t['exit']:.2f}  ({t['reason']})  net Rs {t['net']:+,.0f}"


def cmd_scan(args, c: Config):
    seen, log = set(), Path(args.log)
    step = c.interval * 60
    print(f"Scanning Nifty 50 every {c.interval} min. Signals only: no orders are placed. Ctrl+C to stop.")
    while True:
        now = datetime.now(IST).replace(tzinfo=None)
        if not args.once and (now.weekday() >= 5 or now.hour * 60 + now.minute >= 15 * 60 + 30):
            print("Market closed.")
            return
        if args.once or now.hour * 60 + now.minute >= 9 * 60 + 20:
            data = load(args, c, 5)
            cutoff = now - timedelta(minutes=c.interval)              # drop the candle still forming
            cands = []
            for s, df in data.items():
                df = df[df.index <= cutoff]
                cands += [t for t in find_trades(s, df, c) if t["sig_time"].date() == now.date()]
            taken, _ = run_portfolio(cands, c)
            events = []
            for t in taken:
                events.append((t["sig_time"], "ENTRY", t))
                if t["exit"] is not None:
                    events.append((t["exit_time"], "EXIT", t))
            new = [(ts, k, t) for ts, k, t in sorted(events, key=lambda e: e[0]) if (t["sym"], t["sig_time"], k) not in seen]
            for ts, k, t in new:
                seen.add((t["sym"], t["sig_time"], k))
                late = "  (earlier today)" if now - ts > timedelta(minutes=c.interval + 2) else ""
                line = _fmt_signal(t, k)
                print(f"[{ts:%H:%M}] {line}{late}")
                hdr = not log.exists()
                with log.open("a", newline="") as f:
                    w = csv.writer(f)
                    if hdr:
                        w.writerow(["time", "kind", "symbol", "side", "qty", "price", "stop", "target", "reason", "net"])
                    w.writerow([ts, k, t["sym"], t["side"], t["qty"], round(t["exit"] if k == "EXIT" else t["entry"], 2),
                                round(t["stop"], 2), round(t["target"], 2), t["reason"] if k == "EXIT" else "", t["net"] if k == "EXIT" else ""])
            if not new:
                print(f"[{now:%H:%M}] no new signal  ({len(data)} stocks checked, {len(taken)} trade(s) today)")
        if args.once:
            return
        time.sleep(step - time.time() % step + 20)                    # 20s after each candle closes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["costs", "backtest", "scan"])
    p.add_argument("--source", choices=["yfinance", "groww", "csv", "demo"], default="yfinance")
    p.add_argument("--csv-dir", default="data")
    p.add_argument("--symbols-file", help="text file with one NSE symbol per line (default: built-in Nifty 50)")
    p.add_argument("--days", type=int, default=55, help="backtest length in calendar days")
    p.add_argument("--out", default="trades.csv", help="backtest trade list")
    p.add_argument("--log", default="signals_log.csv", help="live signal log")
    p.add_argument("--once", action="store_true", help="scan: run a single pass and exit")
    p.add_argument("--capital", type=float)
    p.add_argument("--leverage", type=float)
    p.add_argument("--risk-pct", type=float)
    p.add_argument("--rr", type=float)
    p.add_argument("--atr-stop", type=float)
    p.add_argument("--max-trades-day", type=int)
    p.add_argument("--min-edge-mult", type=float)
    p.add_argument("--long-only", action="store_true", default=None)
    a = p.parse_args()
    c = Config()
    for k in ["capital", "leverage", "risk_pct", "rr", "atr_stop", "max_trades_day", "min_edge_mult", "long_only"]:
        if getattr(a, k) is not None:
            setattr(c, k, getattr(a, k))
    {"costs": cmd_costs, "backtest": cmd_backtest, "scan": cmd_scan}[a.command](a, c)


if __name__ == "__main__":
    main()
