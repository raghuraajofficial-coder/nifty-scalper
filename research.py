"""
Research-only tests of published intraday ideas on NSE 5-minute data. Places NO orders.

Every trade is Rs 1 lakh, pays full Groww intraday charges plus 0.03% slippage per order,
and is decided and filled on candle closes, so there is no guessing about what happened
inside a candle.

  N. Noise-area momentum   (Zarattini, Aziz & Barbon 2024, tested there on the S&P 500 ETF)
  M. Market intraday momentum: the first half-hour predicts the last (Gao, Han, Li & Zhou 2018)
  V. VWAP trend-following  (Zarattini & Aziz 2023, tested there on the Nasdaq 100 ETF)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import nifty_scalper as ns

COST = ns.Config()
NOTIONAL = 100_000.0
SLIP = 3 / 1e4
OPEN, LAST = 555, 900                     # 09:15 candle; the 15:00 candle closes at 15:05, before auto square-off


def _trade(day, side, entry, exit_):
    e, x = entry * (1 + side * SLIP), exit_ * (1 - side * SLIP)
    qty = int(NOTIONAL // e)
    if qty < 1 or not np.isfinite(e + x):
        return None
    cost = ns.order_cost(qty * e, "B" if side == 1 else "S", COST) + ns.order_cost(qty * x, "S" if side == 1 else "B", COST)
    return day, side * (x - e) / e * 100, cost / (qty * e) * 100


def _pivots(df: pd.DataFrame):
    """Day x minute-of-day tables of close and VWAP, plus each day's open and previous close."""
    day, m = df.index.normalize(), df.index.hour * 60 + df.index.minute
    close = pd.Series(df["close"].to_numpy(), index=[day, m]).unstack()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    vwap = pd.Series((pv / vv).to_numpy(), index=[day, m]).unstack()
    opn = df["open"].groupby(day).first()
    prev = df["close"].groupby(day).last().shift()
    return close, vwap, opn, prev


def noise_area(df: pd.DataFrame, lookback: int = 14) -> list:
    close, vwap, opn, prev = _pivots(df)
    sigma = close.div(opn, axis=0).sub(1).abs().rolling(lookback).mean().shift()      # typical move from the open, by time of day
    marks = [m for m in close.columns if (m + 5 - OPEN) % 30 == 0 and 580 <= m <= 895]  # decide every half hour, 09:45-15:00
    out = []
    if LAST not in close.columns:
        return out
    for day in close.index:
        o, pc = opn.get(day), prev.get(day)
        if not np.isfinite(o) or not np.isfinite(pc):
            continue
        pos, entry = 0, 0.0
        for m in marks:
            c, s, vw = close.at[day, m], sigma.at[day, m], vwap.at[day, m]
            if not (np.isfinite(c) and np.isfinite(s) and np.isfinite(vw)):
                continue
            ub, lb = max(o, pc) * (1 + s), min(o, pc) * (1 - s)
            if (pos == 1 and c < max(ub, vw)) or (pos == -1 and c > min(lb, vw)):       # trailing stop: band or VWAP
                out.append(_trade(day, pos, entry, c))
                pos = 0
            if pos == 0:
                if c > ub and c > vw:
                    pos, entry = 1, c
                elif c < lb and c < vw:
                    pos, entry = -1, c
        if pos:
            out.append(_trade(day, pos, entry, close.at[day, LAST]))
    return [t for t in out if t]


def half_hour_momentum(df: pd.DataFrame, enter: int, leave: int, market_sign: pd.Series | None = None) -> list:
    """Trade from the close of candle `enter` to the close of candle `leave`, in the direction of the first half-hour."""
    close, _, _, prev = _pivots(df)
    out = []
    if not {580, enter, leave} <= set(close.columns):
        return out
    for day in close.index:
        if market_sign is not None:
            r1 = market_sign.get(day, np.nan)
        else:
            r1 = close.at[day, 580] / prev.get(day) - 1 if np.isfinite(prev.get(day)) else np.nan   # previous close to 09:45
        if not np.isfinite(r1) or r1 == 0:
            continue
        out.append(_trade(day, 1 if r1 > 0 else -1, close.at[day, enter], close.at[day, leave]))
    return [t for t in out if t]


def vwap_trend(df: pd.DataFrame) -> list:
    close, vwap, _, _ = _pivots(df)
    cols = [m for m in close.columns if 560 <= m <= LAST]
    out = []
    for day in close.index:
        pos, entry = 0, 0.0
        for m in cols:
            c, vw = close.at[day, m], vwap.at[day, m]
            if not (np.isfinite(c) and np.isfinite(vw)) or c == vw:
                continue
            want = 1 if c > vw else -1
            if m == LAST:
                break
            if want != pos:                                                           # price crossed VWAP: close and reverse
                if pos:
                    out.append(_trade(day, pos, entry, c))
                pos, entry = want, c
        if pos and np.isfinite(close.at[day, LAST] if LAST in close.columns else np.nan):
            out.append(_trade(day, pos, entry, close.at[day, LAST]))
    return [t for t in out if t]


def summarize(name: str, trades: list) -> dict:
    if not trades:
        return dict(name=name, n=0)
    day = np.array([t[0] for t in trades])
    g, c = np.array([t[1] for t in trades]), np.array([t[2] for t in trades])
    net = g - c
    daily = pd.Series(net).groupby(day).mean()                                        # one number per session, since trades on a day move together
    t_stat = float(daily.mean() / daily.std(ddof=1) * np.sqrt(len(daily))) if len(daily) > 2 and daily.std(ddof=1) > 0 else 0.0
    loss = -net[net <= 0].sum()
    return dict(name=name, n=len(net), days=len(daily), win_pct=round(float((net > 0).mean() * 100), 1),
                gross_pct=round(float(g.mean()), 3), cost_pct=round(float(c.mean()), 3), net_pct=round(float(net.mean()), 3),
                pf=round(float(net[net > 0].sum() / loss), 2) if loss > 0 else None, t_daily=round(t_stat, 2))


def run(data: dict, bees: pd.DataFrame | None) -> list:
    n50 = [s for s in ns.NIFTY50 if s in data]
    rows = []

    def over(name, fn, syms):
        rows.append(summarize(name, [t for s in syms for t in fn(data[s])]))

    if bees is not None and len(bees):
        rows.append(summarize("N1. Noise-area momentum, Nifty ETF (NIFTYBEES)", noise_area(bees)))
    over("N2. Noise-area momentum, Nifty 50 stocks", noise_area, n50)
    over("N3. Noise-area momentum, Nifty 200 stocks", noise_area, list(data))
    if bees is not None and len(bees):
        rows.append(summarize("M1. First half-hour predicts last, Nifty ETF, 15:00-15:25", half_hour_momentum(bees, 895, 920)))
        close, _, _, prev = _pivots(bees)
        sign = (close[580] / prev - 1) if 580 in close.columns else None
        if sign is not None:
            over("M2. Same, Nifty 200 stocks on the market's signal, 15:00-15:25", lambda d: half_hour_momentum(d, 895, 920, sign), list(data))
    over("M3. Same, Nifty 200 stocks on their own signal, 15:00-15:25", lambda d: half_hour_momentum(d, 895, 920), list(data))
    over("M4. Same idea in the margin-tradeable window, 14:35-15:05", lambda d: half_hour_momentum(d, 870, 900), list(data))
    if bees is not None and len(bees):
        rows.append(summarize("V1. VWAP trend-following, Nifty ETF", vwap_trend(bees)))
    over("V2. VWAP trend-following, Nifty 50 stocks", vwap_trend, n50)
    return rows
