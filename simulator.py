#!/usr/bin/env python3
"""
Stockfish Simulator — Self-contained (stdlib only)
Run with: python3 simulator.py
Access: http://localhost:8090
"""
import http.server
import json
import os
import random
import threading
import time
import datetime
from pathlib import Path

PORT = int(os.environ.get("STOCKFISH_PORT", "8090"))
DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

sp500_sample = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "DIS",
    "NFLX", "ADBE", "CRM", "AMD", "INTC", "CSCO", "PEP", "COST", "ABT",
    "TMO", "DHR", "QCOM", "TXN", "LIN", "ORCL", "ACN", "AVGO", "IBM",
]

shared_state = {
    "decision": "HOLD",
    "stock": None,
    "fish_x": 640,
    "fish_y": 360,
    "fish_visible": True,
    "market_open": True,
    "last_trade": None,
    "trades": [],
    "stats": {"total": 0, "buys": 0, "sells": 0},
}

_lock = threading.Lock()
_listeners: list[list] = []


def load_trades():
    path = DATA_DIR / "trades.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_trade(trade):
    trades = load_trades()
    trades.append(trade)
    with open(DATA_DIR / "trades.json", "w") as f:
        json.dump(trades, f, indent=2, default=str)


def notify_listeners(event, data):
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for i, q in enumerate(_listeners):
        try:
            q.append(msg)
        except Exception:
            dead.append(i)
    for i in reversed(dead):
        _listeners.pop(i)


def simulator_loop():
    width, height = 1280, 720
    mid_x = width // 2
    hold_margin = int(width * 0.18)

    pending_decision = None
    pending_start = None
    pending_stock = None
    log_interval = 5.0

    fish_x, fish_y = mid_x, height // 2
    vx, vy = random.choice([-1, 1]) * random.uniform(2, 6), random.choice([-1, 1]) * random.uniform(1, 4)

    while True:
        fish_x += vx
        fish_y += vy

        if fish_x < 40 or fish_x > width - 40:
            vx *= -1
            fish_x = max(40, min(width - 40, fish_x))
        if fish_y < 40 or fish_y > height - 40:
            vy *= -1
            fish_y = max(40, min(height - 40, fish_y))

        if random.random() < 0.02:
            vx += random.uniform(-1.5, 1.5)
            vy += random.uniform(-1.5, 1.5)
            vx = max(-8, min(8, vx))
            vy = max(-6, min(6, vy))

        if fish_x < mid_x - hold_margin:
            decision = "BUY"
        elif fish_x > mid_x + hold_margin:
            decision = "SELL"
        else:
            decision = "HOLD"

        now = datetime.datetime.now()

        now_val = datetime.datetime.now()
        if now_val.weekday() >= 5:
            market_open = False
        else:
            t = now_val.time()
            market_open = datetime.time(9, 30) <= t <= datetime.time(16, 0)

        with _lock:
            shared_state["fish_x"] = int(fish_x)
            shared_state["fish_y"] = int(fish_y)
            shared_state["decision"] = decision
            shared_state["market_open"] = market_open

        if market_open:
            if decision != "HOLD":
                if pending_start is None:
                    pending_start = time.time()
                    pending_decision = decision
                    pending_stock = random.choice(sp500_sample)
            else:
                pending_start = None
                pending_decision = None
                pending_stock = None

            if pending_start and (time.time() - pending_start >= log_interval):
                price = round(random.uniform(20, 400), 2)
                trade = {
                    "decision": pending_decision,
                    "stock": pending_stock,
                    "price": price,
                    "position_x": int(fish_x),
                    "position_y": int(fish_y),
                    "canvas_width": width,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "status": "pending",
                }
                save_trade(trade)

                trades = load_trades()
                buys = sum(1 for t in trades if t.get("decision") == "BUY")
                sells = sum(1 for t in trades if t.get("decision") == "SELL")

                with _lock:
                    shared_state["last_trade"] = trade
                    shared_state["stock"] = pending_stock
                    shared_state["stats"] = {"total": len(trades), "buys": buys, "sells": sells}

                notify_listeners("trade", trade)
                notify_listeners("stats", shared_state["stats"])

                pending_start = None
                pending_decision = None
                pending_stock = None
        else:
            pending_start = None
            pending_decision = None
            pending_stock = None

        notify_listeners("fish", {
            "x": int(fish_x), "y": int(fish_y),
            "decision": decision,
            "market_open": market_open,
            "pending_stock": pending_stock,
            "pending_decision": pending_decision,
        })

        time.sleep(0.05)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0a0a12">
<title>Stockfish — The Fin-fluencer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a12;color:#e2e8f0;font-family:'Courier New',monospace;overflow-x:hidden;-webkit-tap-highlight-color:transparent;touch-action:manipulation}
.layout{display:flex;min-height:100vh;min-height:100dvh}
.sidebar{width:340px;min-width:340px;background:#111122;border-right:1px solid #1e293b;display:flex;flex-direction:column;padding:14px;gap:10px;overflow-y:auto;max-height:100vh;max-height:100dvh}
.main{flex:1;display:flex;align-items:center;justify-content:center;padding:12px;overflow:hidden}
.canvas-wrap{position:relative;width:100%;max-width:100%;aspect-ratio:16/9;max-height:80vh;max-height:80dvh}
canvas{display:block;width:100%;height:100%;border:2px solid #1e293b;border-radius:8px;box-shadow:0 0 40px rgba(34,197,94,.08)}
h1{font-size:20px;color:#22c55e;text-align:center;letter-spacing:2px;text-wrap:nowrap}
h2{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat-card{background:#1a1a2e;border:1px solid #1e293b;border-radius:6px;padding:10px 8px;text-align:center}
.stat-value{font-size:22px;font-weight:bold}
.stat-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-top:2px}
.buy{color:#22c55e}.sell{color:#ef4444}
.trades-list{flex:1;overflow-y:auto;display:flex;flex-direction:column-reverse;gap:6px;-webkit-overflow-scrolling:touch}
.trade-row{display:flex;justify-content:space-between;align-items:center;background:#1a1a2e;border-left:3px solid #334155;border-radius:4px;padding:8px 10px;font-size:12px;gap:8px}
.trade-row[data-decision="BUY"]{border-left-color:#22c55e}
.trade-row[data-decision="SELL"]{border-left-color:#ef4444}
.trade-symbol{font-weight:bold;font-size:14px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trade-decision{font-size:10px;text-transform:uppercase;letter-spacing:1px;flex-shrink:0}
.trade-price{color:#94a3b8;flex-shrink:0}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.status-dot.live{background:#22c55e;box-shadow:0 0 8px #22c55e;animation:pulse 2s infinite}
.status-dot.closed{background:#ef4444;box-shadow:0 0 6px #ef4444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.market-badge{display:inline-block;padding:4px 10px;border-radius:4px;font-size:10px;text-transform:uppercase;letter-spacing:1px}
.market-open{background:#064e3b;color:#22c55e;border:1px solid #22c55e}
.market-closed{background:#450a0a;color:#ef4444;border:1px solid #ef4444}
.last-trade{background:#1a1a2e;border:1px solid #334155;border-radius:6px;padding:10px}
.decision-overlay{font-size:22px;font-weight:bold;text-shadow:0 0 12px rgba(0,0,0,.8)}
.header-row{display:flex;align-items:center;justify-content:space-between}
.header-row h1{font-size:18px}
.mobile-toggle{display:none;background:none;border:1px solid #334155;color:#e2e8f0;font-size:22px;padding:4px 10px;border-radius:6px;cursor:pointer;line-height:1}

@media(max-width:768px){
.layout{flex-direction:column}
.sidebar{width:100%;min-width:100%;max-height:none;border-right:none;border-bottom:1px solid #1e293b;padding:10px;gap:8px;position:relative}
.main{flex:1;padding:8px}
.canvas-wrap{max-height:45vh;max-height:45dvh}
h1{font-size:16px}
.stat-value{font-size:18px}
.trade-row{font-size:11px;padding:6px 8px}
.trade-symbol{font-size:12px}
.mobile-toggle{display:block}
.sidebar .stats-section,.sidebar .trades-section,.sidebar .last-trade,.sidebar .footer-note{display:none}
.sidebar.expanded .stats-section,.sidebar.expanded .trades-section,.sidebar.expanded .last-trade,.sidebar.expanded .footer-note{display:block}
.sidebar.expanded{max-height:55vh;overflow-y:auto}
}
</style>
</head>
<body>
<div class="layout">
<div class="sidebar" id="sidebar">
<div class="header-row">
<h1>STOCKFISH</h1>
<button class="mobile-toggle" id="toggleBtn" aria-label="Toggle stats">☰</button>
</div>
<div><span id="marketBadge" class="market-badge market-closed"><span class="status-dot closed"></span> CHECKING...</span></div>
<div class="stats-section">
<h2>Trading Stats</h2>
<div class="stats-grid">
<div class="stat-card"><div class="stat-value" id="total">0</div><div class="stat-label">Total</div></div>
<div class="stat-card"><div class="stat-value buy" id="buys">0</div><div class="stat-label">Buys</div></div>
<div class="stat-card"><div class="stat-value sell" id="sells">0</div><div class="stat-label">Sells</div></div>
<div class="stat-card"><div class="stat-value" id="ratio">—</div><div class="stat-label">Ratio</div></div>
</div>
</div>
<div id="lastBlock" class="last-trade" style="display:none">
<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px">Last Trade</div>
<div style="font-size:16px;font-weight:bold" id="lastInfo">—</div>
</div>
<div class="trades-section" style="flex:1;min-height:0;display:flex;flex-direction:column">
<h2>Recent Trades</h2>
<div class="trades-list" id="tradeList"><div style="color:#475569;text-align:center;padding:20px">Waiting...</div></div>
</div>
<div class="footer-note" style="font-size:10px;color:#475569;text-align:center;margin-top:auto;padding-top:8px">Simulator Mode | No webcam</div>
</div>
<div class="main">
<div class="canvas-wrap">
<canvas id="canvas" width="1280" height="720"></canvas>
<div id="overlay" class="decision-overlay" style="position:absolute;bottom:12px;left:12px;color:#fff">HOLD</div>
<div id="pendingBadge" style="position:absolute;top:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.7);padding:4px 12px;border-radius:4px;font-size:12px;display:none"></div>
</div>
</div>
</div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = 1280, H = 720, MID = W / 2;

let fish = {x: MID, y: H/2, decision: 'HOLD', market_open: true};
let pending = null;
let trades = [];

const toggleBtn = document.getElementById('toggleBtn');
const sidebar = document.getElementById('sidebar');
let sidebarOpen = window.innerWidth > 768;

toggleBtn.addEventListener('click', () => {
    sidebarOpen = !sidebarOpen;
    if (sidebarOpen) sidebar.classList.add('expanded');
    else sidebar.classList.remove('expanded');
});

window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        sidebar.classList.remove('expanded');
        sidebarOpen = true;
    }
});

function draw() {
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // Divider
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(MID, 0);
    ctx.lineTo(MID, H);
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.font = 'bold 16px "Courier New"';
    ctx.fillStyle = 'rgba(34,197,94,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('BUY', MID/2, 45);
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    ctx.fillText('SELL', MID + MID/2, 45);

    // Fish
    let fx = fish.x, fy = fish.y;
    let color = fish.decision === 'BUY' ? '#22c55e' : fish.decision === 'SELL' ? '#ef4444' : '#fbbf24';

    // Glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 20;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(fx, fy, 16, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Eye
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(fx + 5, fy - 3, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(fx + 6, fy - 4, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Tail
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(fx - 14, fy);
    ctx.lineTo(fx - 26, fy - 10);
    ctx.moveTo(fx - 14, fy);
    ctx.lineTo(fx - 26, fy + 10);
    ctx.stroke();

    requestAnimationFrame(draw);
}

const evtSource = new EventSource('/stream');

evtSource.addEventListener('fish', e => {
    const d = JSON.parse(e.data);
    fish = d;
    document.getElementById('overlay').textContent = d.decision;
    document.getElementById('overlay').style.color =
        d.decision === 'BUY' ? '#22c55e' : d.decision === 'SELL' ? '#ef4444' : '#fff';

    const badge = document.getElementById('marketBadge');
    if (d.market_open) {
        badge.className = 'market-badge market-open';
        badge.innerHTML = '<span class="status-dot live"></span> MARKET OPEN';
    } else {
        badge.className = 'market-badge market-closed';
        badge.innerHTML = '<span class="status-dot closed"></span> MARKET CLOSED';
    }

    const pb = document.getElementById('pendingBadge');
    if (d.pending_stock) {
        pb.style.display = 'block';
        const sec = '...';
        pb.innerHTML = `WAITING: <b>${d.pending_stock}</b> ${d.pending_decision}`;
        pb.style.color = d.pending_decision === 'BUY' ? '#22c55e' : '#ef4444';
    } else {
        pb.style.display = 'none';
    }

    pending = d;
});

evtSource.addEventListener('trade', e => {
    const t = JSON.parse(e.data);
    trades.unshift(t);
    if (trades.length > 50) trades.length = 50;
    renderTrades();

    document.getElementById('lastBlock').style.display = 'block';
    const cls = t.decision === 'BUY' ? 'buy' : 'sell';
    document.getElementById('lastInfo').innerHTML =
        `<span class="${cls}">${t.decision}</span> ${t.stock} @ $${t.price}`;
});

evtSource.addEventListener('stats', e => {
    const s = JSON.parse(e.data);
    document.getElementById('total').textContent = s.total;
    document.getElementById('buys').textContent = s.buys;
    document.getElementById('sells').textContent = s.sells;
    document.getElementById('ratio').textContent =
        s.sells > 0 ? (s.buys / s.sells).toFixed(2) : '-';
});

function renderTrades() {
    const el = document.getElementById('tradeList');
    el.innerHTML = trades.slice(0, 50).map(t => {
        const cls = t.decision === 'BUY' ? 'buy' : 'sell';
        return `<div class="trade-row" data-decision="${t.decision}">
            <span class="trade-symbol">${t.stock}</span>
            <span class="trade-decision ${cls}">${t.decision}</span>
            <span class="trade-price">$${t.price}</span>
        </div>`;
    }).join('');
}

draw();
</script>
</body>
</html>"""


class SimHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html(INDEX_HTML)
        elif self.path == "/stream":
            self._serve_sse()
        elif self.path == "/api/stats":
            self._serve_json(shared_state["stats"])
        elif self.path == "/api/trades":
            self._serve_json(load_trades()[-50:])
        elif self.path == "/api/state":
            self._serve_json({
                "decision": shared_state["decision"],
                "market_open": shared_state["market_open"],
                "fish_x": shared_state["fish_x"],
                "fish_y": shared_state["fish_y"],
                "last_trade": shared_state["last_trade"],
                "stats": shared_state["stats"],
            })
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/reset":
            with _lock:
                shared_state["stats"] = {"total": 0, "buys": 0, "sells": 0}
                shared_state["last_trade"] = None
                shared_state["trades"] = []
            trades_path = DATA_DIR / "trades.json"
            if trades_path.exists():
                trades_path.unlink()
            self._serve_json({"ok": True})
        else:
            self.send_error(404)

    def _serve_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q: list = []
        _listeners.append(q)

        try:
            while True:
                if q:
                    msg = q.pop(0)
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                else:
                    time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            try:
                _listeners.remove(q)
            except ValueError:
                pass

    def log_message(self, format, *args):
        pass


def main():
    print(f"\n  Stockfish Simulator")
    print(f"  Dashboard: http://localhost:{PORT}")
    print(f"  API:       http://localhost:{PORT}/api/stats")
    print(f"  Press Ctrl+C to stop\n")

    sim_thread = threading.Thread(target=simulator_loop, daemon=True)
    sim_thread.start()

    server = http.server.HTTPServer(("0.0.0.0", PORT), SimHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
