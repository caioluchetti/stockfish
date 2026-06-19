import json
import logging
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

logger = logging.getLogger(__name__)

app = Flask(__name__)

_frame_buffer = None
_frame_lock = threading.Lock()
_config_ref = None
_data_dir = Path("data")


def set_frame_buffer(buffer_ref, lock):
    global _frame_buffer, _frame_lock
    _frame_buffer = buffer_ref
    _frame_lock = lock


def set_data_dir(path):
    global _data_dir
    _data_dir = Path(path)


def _read_trades():
    trades_file = _data_dir / "trades.json"
    if not trades_file.exists():
        return []
    try:
        with open(trades_file) as f:
            return json.load(f)
    except Exception:
        return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trades")
def api_trades():
    trades = _read_trades()
    return jsonify(trades[-50:])


@app.route("/api/stats")
def api_stats():
    trades = _read_trades()
    buys = [t for t in trades if t.get("decision") == "BUY"]
    sells = [t for t in trades if t.get("decision") == "SELL"]
    return jsonify({
        "total_trades": len(trades),
        "buys": len(buys),
        "sells": len(sells),
        "last_trade": trades[-1] if trades else None,
    })


@app.route("/api/market")
def api_market():
    from src.market import Market
    from src.config import Config

    market = Market(Config())
    is_open = market.is_market_open()

    import datetime
    import pytz
    est = pytz.timezone("US/Eastern")
    now = datetime.datetime.now(est)

    if is_open:
        message = "Market is open"
    elif now.weekday() >= 5:
        message = "Weekend — closed"
    else:
        message = "Outside trading hours (9:30 AM – 4:00 PM EST)"

    return jsonify({
        "market_open": is_open,
        "message": message,
    })


@app.route("/api/live")
def api_live():
    with _frame_lock:
        if _frame_buffer is None:
            return jsonify({"decision": "HOLD", "market_open": False})

        return jsonify({
            "decision": _frame_buffer.get("decision", "HOLD"),
            "market_open": _frame_buffer.get("market_open", False),
        })


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _frame_lock:
                if _frame_buffer is None or _frame_buffer.get("frame") is None:
                    continue
                frame = _frame_buffer["frame"].copy()

            import cv2
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start_server(host="0.0.0.0", port=5000):
    def run():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    logger.info("Web UI started at http://%s:%d", host, port)
    return thread
