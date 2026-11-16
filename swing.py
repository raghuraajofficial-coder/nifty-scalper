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


def load(source: str, period: str = "12y", volume: bool = False):
    syms = orb.NIFTY200
    if source == "demo":
        rng = np.random.default_rng(5)
        idx = pd.bdate_range(end=datetime.now().date(), periods=3000)
        drift = rng.normal(0.0004, 0.0004, len(syms))
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, 0.018, (len(idx), len(syms))), axis=0)), index=idx, columns=syms)
        open_ = close.shift().fillna(close) * (1 + rng.normal(0, 0.004, close.shape))
        vol = pd.DataFrame(rng.lognormal(12, 1, close.shape) * rng.lognormal(0, 1, len(syms)), index=idx, columns=syms)
        return (close, open_, close.mean(axis=1), vol) if volume else (close, open_, close.mean(axis=1))
    import yfinance as yf
    closes, opens, vols = {}, {}, {}
    for i in range(0, len(syms), 50):
        raw = yf.download([f"{s}.NS" for s in syms[i:i + 50]], period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for s in syms[i:i + 50]:
            try:
                df = raw[f"{s}.NS"].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) > 300:
                closes[s], opens[s], vols[s] = df["Close"], df["Open"], df["Volume"]
        time.sleep(1.5)
    nifty = yf.download("^NSEI", period=period, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
    close, open_, vol = pd.DataFrame(closes), pd.DataFrame(opens), pd.DataFrame(vols)
    for x in (close, open_, nifty, vol):
        x.index = pd.DatetimeIndex(x.index).tz_localize(None)
    close = close[close.index.isin(nifty.index)]
    out = (close, open_.reindex(close.index), nifty.reindex(close.index).ffill())
    return out + (vol.reindex(close.index),) if volume else out


# Broad sector groups, assigned by hand, for the sector-cap screen. A stock not listed here is its own group.
SECTORS = {
    "Banks": "AUBANK AXISBANK BANKBARODA BANKINDIA MAHABANK CANBK FEDERALBNK HDFCBANK ICICIBANK IDFCFIRSTB INDIANB INDUSINDBK KOTAKBANK PNB SBIN UNIONBANK YESBANK",
    "Financial services": "360ONE ABCAPITAL BSE BAJFINANCE BAJAJFINSV BAJAJHLDNG CHOLAFIN GROWW HDFCAMC HDFCLIFE ICICIGI ICICIAMC IRFC IREDA JIOFIN LTF LICHSGFIN LICI M&MFIN MFSL MOTILALOFS MCX MUTHOOTFIN PAYTM POLICYBZR PFC RECLTD SBICARD SBILIFE SHRIRAMFIN TATACAP",
    "Capital goods": "ABB APLAPOLLO APARINDS ASTRAL BDL BEL BHEL CGPOWER COCHINSHIP CUMMINSIND GVT&D HAL POWERINDIA KEI MAZDOCK POLYCAB PREMIERENE ENRIN SIEMENS SUPREMEIND SUZLON WAAREEENER",
    "Construction": "LT RVNL",
    "Automobiles": "ASHOKLEY BAJAJ-AUTO BHARATFORG BOSCHLTD EICHERMOT EXIDEIND HEROMOTOCO HYUNDAI MRF M&M MARUTI MOTHERSON TVSMOTOR TMCV TMPV TIINDIA",
    "Oil, gas and coal": "ATGL BPCL COALINDIA GAIL HINDPETRO IOC ONGC OIL RELIANCE",
    "Power": "ADANIENSOL ADANIGREEN ADANIPOWER JSWENERGY NHPC NLCINDIA NTPC POWERGRID TATAPOWER",
    "Metals and mining": "ADANIENT HINDALCO HINDCOPPER HINDZINC JSWSTEEL JINDALSTEL NMDC NATIONALUM SAIL TATASTEEL VEDL",
    "Cement": "AMBUJACEM GRASIM ULTRACEMCO",
    "Healthcare": "APOLLOHOSP AUROPHARMA BIOCON CIPLA DIVISLAB DRREDDY FORTIS GLENMARK LAURUSLABS LUPIN MANKIND MAXHEALTH SUNPHARMA TORNTPHARM ZYDUSLIFE",
    "Information technology": "COFORGE HCLTECH INFY LTM MPHASIS OFSS PERSISTENT TCS TECHM WIPRO",
    "Consumer staples": "BRITANNIA COLPAL DABUR GODFRYPHLP GODREJCP HINDUNILVR ITC MARICO NESTLEIND PATANJALI RADICO TATACONSUM UNITDSPR VBL",
    "Consumer durables": "ASIANPAINT BLUESTARCO DIXON HAVELLS KALYANKJIL LGEINDIA PAGEIND TITAN VOLTAS",
    "Consumer services": "DMART ETERNAL NYKAA INDHOTEL IRCTC NAUKRI JUBLFOOD LENSKART MEESHO SWIGGY TRENT VMM",
    "Property": "DLF GODREJPROP LODHA OBEROIRLTY PHOENIXLTD PRESTIGE",
    "Chemicals": "PIIND PIDILITIND SRF UPL SOLARINDS",
    "Telecom": "BHARTIARTL INDUSTOWER TATACOMM IDEA",
    "Transport": "ADANIPORTS CONCOR GMRAIRPORT INDIGO",
}
SECTOR = {sym: name for name, syms in SECTORS.items() for sym in syms.split()}


def screens(close, vol, c: "Cfg") -> dict:
    """Yes/no tables (date x stock): may this stock be bought on this day? One table per screen.
    Every threshold was fixed before the results were seen, and each uses only data up to that day."""
    r = np.log(close).diff()
    w = c.look - c.skip                                     # the same window the momentum ranking measures
    total = r.rolling(w).sum().shift(c.skip)
    top2 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for s in close.columns:                                  # sum of the two biggest daily gains inside the window
        x = r[s].fillna(0).to_numpy()
        if len(x) >= w:
            win = np.lib.stride_tricks.sliding_window_view(x, w)
            top2.iloc[w - 1:, top2.columns.get_loc(s)] = np.partition(win, w - 2, axis=1)[:, -2:].sum(axis=1)
    top2 = top2.shift(c.skip)
    up_share = (r > 0).where(r.notna()).rolling(w).mean().shift(c.skip)
    sd = r.rolling(63).std()
    out = {"jump": ~((total > 0) & (top2 > 0.5 * total)),
           "smooth": up_share.ge(up_share.median(axis=1), axis=0),
           "calm": sd.le(sd.quantile(0.8, axis=1), axis=0)}
    if vol is not None:
        traded = (close * vol).rolling(63).median()
        out["liquid"] = traded.ge(traded.quantile(0.2, axis=1), axis=0)
    return out


def simulate(close, open_, nifty, c: Cfg, start: int, ok=None, cap: int = 0):
    dates = close.index
    mom = close.shift(c.skip) / close.shift(c.look) - 1
    ma = nifty.rolling(200).mean()
    last = close.ffill()
    cash, pos, pending, stops = c.capital, {}, None, set()
    fy_gain, carry, tax_paid, charges, n_trades, wins = 0.0, 0.0, 0.0, 0.0, 0, 0
    curve, closed, vetoes, short = [], [], 0, 0
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
                new = []
                for s in [s for s in rank[:c.buffer] if s not in keep]:     # best-ranked first, never below the top 20
                    if len(keep) + len(new) >= c.top_n:
                        break
                    if ok is not None and not bool(ok.at[d, s]):
                        vetoes += 1
                        continue
                    if cap and sum(SECTOR.get(x, x) == SECTOR.get(s, s) for x in keep + new) >= cap:
                        vetoes += 1
                        continue
                    new.append(s)
                short += c.top_n - len(keep) - len(new)                     # slots left empty because too few passed
                pending = keep + new
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
                         avg_hold_weeks=round(float(held.mean()) / 5, 1), median_hold_weeks=round(float(np.median(held)) / 5, 1),
                         vetoes=vetoes, empty_slots=short), closed


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


# Mechanical screens tested on top of the featured version. Each can only block a NEW purchase; the next-ranked
# stock inside the top 20 takes its place. They stand in for the usual reasons a researcher would reject a stock.
SCREENS = [
    ("jump", "Skip one-off jumps", "Skips a stock when its two biggest single-day gains make up more than half of its rise over the ranking period. That pattern is typical of a takeover offer or a single piece of news."),
    ("smooth", "Prefer steady risers", "Skips a stock if it rose on fewer days than the typical Nifty 200 stock did over the ranking period. A steady climb is thought to last longer than a few big bursts."),
    ("calm", "Skip the most volatile", "Skips a stock if its daily swings over the past three months are among the wildest fifth of the 200."),
    ("liquid", "Skip thinly traded", "Skips a stock if the rupee value traded in it each day over the past three months is among the lowest fifth of the 200."),
    ("sector", "At most two per sector", "Skips a stock if two holdings are already from the same sector, using 18 broad sector groups."),
]


def main(source: str):
    close, open_, nifty, vol = load(source, volume=True)
    start = Cfg().look + 2
    rows, feat = [], None
    for name, over in VARIANTS:
        eq, pre, extra, log = simulate(close, open_, nifty, replace(Cfg(), **over), start)
        rows.append(dict(name=name, after_tax=perf(eq), before_tax_cagr=perf(pre)["cagr"], **extra))
        if name.startswith(FEATURED):
            feat, feat_log = eq, log
    feat_cfg = replace(Cfg(), **next(o for n, o in VARIANTS if n.startswith(FEATURED)))
    tables = screens(close, vol, feat_cfg)
    half = close.index[start + (len(close.index) - start) // 2]
    def halves(eq):                                        # return per year in each half of the test, to see if a gain is consistent
        a, b = eq[eq.index <= half], eq[eq.index >= half]
        g = lambda x: round(((x.iloc[-1] / x.iloc[0]) ** (365.25 / max(1, (x.index[-1] - x.index[0]).days)) - 1) * 100, 1)
        return [float(g(a)), float(g(b))]
    tested = []
    for key, title, what in SCREENS:
        eq, pre, extra, _ = simulate(close, open_, nifty, feat_cfg, start, ok=tables.get(key), cap=2 if key == "sector" else 0)
        tested.append(dict(key=key, name=title, what=what, after_tax=perf(eq), halves=halves(eq), **extra))
    span = close.index[start:]
    bench = {"Nifty 50 index, buy and hold (dividends excluded)": nifty.loc[span] / nifty.loc[span[0]] * 100_000,
             "Equal weight of the same 200 stocks, no selection": (1 + close.pct_change(fill_method=None).loc[span].mean(axis=1).fillna(0)).cumprod() * 100_000}
    out = dict(generated=datetime.now(ns.IST).strftime("%Y-%m-%d %H:%M"), source=source, stocks=close.shape[1],
               first_day=str(span[0])[:10], last_day=str(span[-1])[:10], capital=100_000, rows=rows,
               benchmarks=[dict(name=k, **perf(v)) for k, v in bench.items()],
               screens=dict(split=str(half)[:10], base_halves=halves(feat) if feat is not None else None, tested=tested))
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
    for t in tested:
        a = t["after_tax"]
        print(f"{'  + ' + t['name']:<74}{a['cagr']:>7}{'':>8}{a['maxdd']:>8}{a['worst_year']:>7}{a['best_year']:>7}{t['trades']:>8}{t['win_pct']:>6}"
              f"{t['charges']:>9,}{t['tax']:>9,}{a['final']:>11,}  blocked {t['vetoes']}, empty slots {t['empty_slots']}, halves {t['halves']}")
    for b in out["benchmarks"]:
        print(f"{b['name']:<74}{b['cagr']:>7}{'':>8}{b['maxdd']:>8}{b['worst_year']:>7}{b['best_year']:>7}{'':>32}{b['final']:>11,}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "yfinance")
