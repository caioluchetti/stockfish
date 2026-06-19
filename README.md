# Stockfish — The Fin-fluencer Trading Bot

A creative computer-vision trading bot. Point your webcam at a physical object (the "fish") — if it swims **LEFT** of center, it's a **BUY** signal. If it swims **RIGHT**, it's a **SELL**. Hold it still in the middle for **HOLD**.

Decisions are logged to Firebase Firestore (or locally as JSON fallback). Trades only execute during US market hours (Mon-Fri, 9:30 AM – 4:00 PM EST).

## Setup

```bash
# Clone
git clone https://github.com/caioluchetti/stockfish.git
cd stockfish

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Firebase credentials
```

### Environment Variables

| Variable | Description |
|---|---|
| `APP_ID` | Your Firebase/ePanel app ID |
| `USER_ID` | Your user ID |
| `CREDENTIAL_PATH` | Path to Firebase service account JSON key |

## Usage

### Live mode (with webcam)
```bash
python main.py
```

### Simulate mode (no webcam needed — random positions)
```bash
python main.py --simulate
```

### Dry run (no Firebase writes)
```bash
python main.py --dry-run
```

### All options
```bash
python main.py --help
```

| Flag | Description |
|---|---|
| `--simulate` | Skip webcam, generate random fish positions |
| `--dry-run` | Disable Firebase writes |
| `--no-firebase` | Skip Firebase entirely, use local JSON |
| `--debug` | Enable debug logging |
| `--camera N` | Camera device ID (default: 0) |

## Architecture

```
stockfish/
├── main.py                  # Entry point + main loop
├── src/
│   ├── config.py            # All configuration (dataclasses)
│   ├── camera.py            # Webcam capture + background subtractor
│   ├── detector.py          # Fish detection + BUY/SELL/HOLD decision
│   ├── market.py            # Market hours, tickers, price fetching
│   ├── database.py          # Firebase + local JSON fallback
│   └── ui.py                # PIL overlay rendering
├── market_data/
│   ├── sp500_table.csv      # S&P 500 ticker list
│   └── marketdata_api.py    # Standalone: fetch all prices at once
└── data/
    └── trades.json          # Local trade log (fallback)
```
