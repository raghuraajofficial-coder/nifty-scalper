#!/usr/bin/env python3
"""
Hosted paper-trading server for the Nifty 50 scalper.

Runs the scanner every 5 minutes during market hours, records every paper trade in
SQLite, and serves a dashboard (trade list + P&L calendar). Places NO real orders.

Settings (environment variables)
  SOURCE          yfinance | groww | csv | demo      (default yfinance)
  CAPITAL         starting paper capital             (default 100000)
  BACKFILL_DAYS   days of backtest shown in the Backtest tab (default 30, 0 = off)
  DASH_PASSWORD   if set, the dashboard asks for this password (any username)
  PORT            web port                           (default 8000)
  DATA_DIR        where trades.db is kept            (default: this folder)
  GROWW_API_TOKEN only for SOURCE=groww
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import nifty_scalper as ns

HERE = Path(__file__).parent
DB = Path(os.environ.get("DATA_DIR", HERE)) / "trades.db"
SOURCE = os.environ.get("SOURCE", "yfinance")
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "30"))
PASSWORD = os.environ.get("DASH_PASSWORD", "")
ARGS = SimpleNamespace(source=SOURCE, symbols_file=os.environ.get("SYMBOLS_FILE"), csv_dir=os.environ.get("CSV_DIR", "data"))
CFG = ns.Config(capital=float(os.environ.get("CAPITAL", "100000")))
COLS = ["sym", "side", "qty", "sig_time", "entry_time", "entry", "stop", "target", "exit_time", "exit", "reason", "gross", "cost", "net"]
LOCK = threading.Lock()


def now_ist() -> datetime:
    return datetime.now(ns.IST).replace(tzinfo=None)


def db():
    con = sqlite3.connect(DB)
    con.execute(f"CREATE TABLE IF NOT EXISTS trades (mode TEXT, {', '.join(COLS)}, PRIMARY KEY (mode, sym, sig_time))")
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    return con


def set_meta(**kv):
    with LOCK, db() as con:
        con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in kv.items()])


def save(mode: str, trades: list[dict], day: str | None = None):
    """Replace one day's trades (live) or the whole set (backtest) with a fresh, consistent result."""
    rows = [[mode] + [str(t[c]) if c.endswith("_time") and t[c] is not None else
                      (None if t[c] is None else (float(t[c]) if not isinstance(t[c], (str, int)) else t[c])) for c in COLS]
            for t in trades]
    with LOCK, db() as con:
        if day:
            con.execute("DELETE FROM trades WHERE mode = ? AND substr(sig_time, 1, 10) = ?", (mode, day))
        else:
            con.execute("DELETE FROM trades WHERE mode = ?", (mode,))
        con.executemany(f"INSERT OR REPLACE INTO trades VALUES ({', '.join('?' * (len(COLS) + 1))})", rows)


def realized_before(day: str) -> float:
    with LOCK, db() as con:
        return con.execute("SELECT COALESCE(SUM(net), 0) FROM trades WHERE mode = 'paper' AND substr(sig_time, 1, 10) < ?", (day,)).fetchone()[0]


def scan_once():
    now = now_ist()
    today = now.strftime("%Y-%m-%d")
    cfg = replace(CFG, capital=CFG.capital + realized_before(today))
    data = ns.load(ARGS, cfg, 5)
    if not data:
        set_meta(last_scan=now.strftime("%Y-%m-%d %H:%M"), stocks=0, error="No market data received")
        return
    cutoff = now - timedelta(minutes=cfg.interval)            # ignore the candle still forming
    cands = []
    for s, df in data.items():
        cands += [t for t in ns.find_trades(s, df[df.index <= cutoff], cfg) if str(t["sig_time"])[:10] == today]
    taken, _ = ns.run_portfolio(cands, cfg)
    save("paper", taken, day=today)
    set_meta(last_scan=now.strftime("%Y-%m-%d %H:%M"), stocks=len(data), error="")


def backfill():
    if BACKFILL_DAYS <= 0:
        return
    try:
        data = ns.load(ARGS, CFG, BACKFILL_DAYS)
        cands = [t for s, df in data.items() for t in ns.find_trades(s, df, CFG)]
        taken, _ = ns.run_portfolio(cands, CFG)
        save("backtest", [t for t in taken if t["net"] is not None])
        set_meta(backfill=now_ist().strftime("%Y-%m-%d %H:%M"))
    except Exception:                                        # noqa: BLE001
        set_meta(error="Backtest failed: " + traceback.format_exc(limit=1).splitlines()[-1])


def market_open(now: datetime) -> bool:
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and 9 * 60 + 15 <= m < 15 * 60 + 30


def scanner_loop():
    backfill()
    step = CFG.interval * 60
    while True:
        now = now_ist()
        m = now.hour * 60 + now.minute
        if now.weekday() < 5 and 9 * 60 + 20 <= m < 15 * 60 + 40:      # runs a little past close to book square-offs
            try:
                scan_once()
            except Exception:                                # noqa: BLE001
                set_meta(error=traceback.format_exc(limit=1).splitlines()[-1])
        time.sleep(step - time.time() % step + 20)


def state() -> dict:
    with LOCK, db() as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM trades ORDER BY sig_time")]
        meta = dict(con.execute("SELECT k, v FROM meta").fetchall())
    now = now_ist()
    return {
        "status": {"now": now.strftime("%Y-%m-%d %H:%M"), "market_open": market_open(now), "source": SOURCE,
                   "capital": CFG.capital, "backfill_days": BACKFILL_DAYS, "last_scan": meta.get("last_scan", ""),
                   "stocks": meta.get("stocks", ""), "error": meta.get("error", "")},
        "trades": {"paper": [r for r in rows if r["mode"] == "paper"],
                   "backtest": [r for r in rows if r["mode"] == "backtest"]},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if PASSWORD:
            got = self.headers.get("Authorization", "")
            try:
                ok = got.startswith("Basic ") and base64.b64decode(got[6:]).decode().split(":", 1)[1] == PASSWORD
            except Exception:                                # noqa: BLE001
                ok = False
            if not ok:
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="Nifty scalper"')
                self.end_headers()
                return
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (HERE / "docs" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, json.dumps(state()).encode(), "application/json")
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if not PASSWORD:
        print("WARNING: DASH_PASSWORD is not set, so anyone with the address can see the dashboard.")
    threading.Thread(target=scanner_loop, daemon=True).start()
    print(f"Dashboard on port {port} | source={SOURCE} | paper capital Rs {CFG.capital:,.0f} | no real orders are placed")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
