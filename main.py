#!/usr/bin/env python3
import argparse
import logging
import random
import time
import signal
import sys
import threading

import cv2

from src.config import Config
from src.camera import Camera
from src.detector import Detector
from src.market import Market
from src.database import Database
from src.ui import UI
from src.broker.alpaca import AlpacaBroker, NoOpBroker


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stockfish — The Fin-fluencer Trading Bot"
    )
    parser.add_argument("--dry-run", action="store_true", help="Disable Firebase writes")
    parser.add_argument("--simulate", action="store_true", help="Simulate fish positions (no webcam)")
    parser.add_argument("--no-firebase", action="store_true", help="Disable Firebase entirely, use local storage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--broker", choices=["alpaca", "none"], default="none", help="Broker for order execution")
    parser.add_argument("--web", action="store_true", help="Start web dashboard (http://localhost:5000)")
    parser.add_argument("--web-port", type=int, default=5000, help="Web dashboard port")
    return parser.parse_args()


def simulate_position(frame_width, frame_height):
    modes = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    decision = random.choices(modes, weights=weights, k=1)[0]

    mid_x = frame_width // 2
    mid_y = frame_height // 2

    if decision == "BUY":
        x = random.randint(30, mid_x - 80)
    elif decision == "SELL":
        x = random.randint(mid_x + 80, frame_width - 30)
    else:
        x = random.randint(mid_x - 50, mid_x + 50)

    y = random.randint(50, frame_height - 50)
    return True, x, y, decision


def create_blank_frame(width=1280, height=720):
    import numpy as np
    return np.full((height, width, 3), (20, 20, 30), dtype=np.uint8)


def main():
    args = parse_args()
    setup_logging(debug=args.debug)
    logger = logging.getLogger("stockfish")

    config = Config(
        dry_run=args.dry_run,
        simulate=args.simulate,
        no_firebase=args.no_firebase or args.dry_run,
    )
    config.camera.device_id = args.camera

    logger.info("Stockfish starting up... Mode: %s",
                "SIMULATE" if config.simulate else "LIVE")

    market = Market(config)
    database = Database(config)
    ui = UI(config.ui)
    detector = Detector(config.detector)

    if args.broker == "alpaca":
        config.alpaca.enabled = True
        broker = AlpacaBroker(config.alpaca)
    else:
        broker = NoOpBroker()

    web_frame_buffer = {"frame": None}
    web_lock = threading.Lock()

    if args.web:
        try:
            from src.web.app import set_frame_buffer, set_data_dir, start_server

            set_frame_buffer(web_frame_buffer, web_lock)
            set_data_dir(config.data_dir)
            start_server(port=args.web_port)
        except ImportError as e:
            logger.error("Flask not installed. Run: pip install flask")
            logger.error("Web UI disabled.")
        except Exception as e:
            logger.error("Web UI failed to start: %s", e)

    camera = None
    if not config.simulate:
        camera = Camera(config.camera)
        try:
            camera.start()
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.info("Simulate mode — no webcam needed. Press 'q' to quit.")

    pending_decision = None
    pending_start_time = None
    current_stock = None

    running = True

    def shutdown(sig, frame):
        nonlocal running
        logger.info("Shutting down...")
        running = False

    signal.signal(signal.SIGINT, shutdown)

    try:
        while running:
            loop_start = time.time()

            if config.simulate:
                frame = create_blank_frame()
                clean_mask = None
                found_fish, fish_x, fish_y, current_decision = simulate_position(
                    frame.shape[1], frame.shape[0]
                )
            else:
                frame = camera.read_frame()
                if frame is None:
                    break

                fg_mask = camera.apply_background_subtraction(frame)
                clean_mask, contours = detector.process_mask(fg_mask)
                found_fish, fish_x, fish_y, current_decision = detector.detect_fish(
                    contours, frame.shape[1]
                )

            market_is_open = market.is_market_open()

            if market_is_open:
                if current_decision != "HOLD":
                    if pending_start_time is None:
                        pending_start_time = time.time()
                        pending_decision = current_decision
                        current_stock = market.pick_random_ticker()
                        logger.info("WAITING %s — %s for %.0fs",
                                    pending_decision, current_stock, config.trading.log_interval)
                else:
                    pending_start_time = None
                    pending_decision = None
                    current_stock = None

                if pending_start_time and (time.time() - pending_start_time >= config.trading.log_interval):
                    price = market.get_price(current_stock)

                    decision_data = {
                        "decision": pending_decision,
                        "stock": current_stock,
                        "price": price,
                        "position_x": fish_x,
                        "position_y": fish_y,
                        "canvas_width": frame.shape[1],
                    }

                    database.log_trade(decision_data)
                    logger.info("TRADE LOGGED — %s | %s | $%s",
                                pending_decision, current_stock, price)

                    if config.alpaca.enabled:
                        order = broker.place_order(current_stock, pending_decision, qty=1)
                        decision_data["order_id"] = order.status if order.status != "failed" else None

                    pending_start_time = None
                    pending_decision = None
                    current_stock = None
            else:
                pending_start_time = None
                pending_decision = None
                current_stock = None

            frame = ui.render(
                frame,
                current_decision if market_is_open else "HOLD",
                pending_decision,
                current_stock,
                found_fish,
                fish_x,
                fish_y,
                market_is_open,
            )

            with web_lock:
                web_frame_buffer["frame"] = frame
                web_frame_buffer["market_open"] = market_is_open
                web_frame_buffer["decision"] = current_decision

            if not config.simulate and fg_mask is not None:
                cv2.imshow("Debug Mask - What the Bot Sees", clean_mask)
            cv2.imshow("The Fin-fluencer BOT (Press 'q' to quit)", frame)

            key = cv2.waitKey(max(1, int(1000 / config.camera.fps) - int((time.time() - loop_start) * 1000))) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested.")
                running = False
            elif key == ord('c') and camera is not None:
                camera.recalibrate()

    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        logger.info("Stockfish terminated.")


if __name__ == "__main__":
    main()
