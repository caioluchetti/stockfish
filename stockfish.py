import cv2
import numpy as np
import time
import datetime
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import random
import yfinance as yf
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import pytz

stocks = pd.read_csv(r'market_data\sp500_table.csv')
tickers = stocks['Symbol'].tolist()


load_dotenv()



APP_ID = os.getenv('APP_ID')
USER_ID = os.getenv('USER_ID')

# --- USER CONFIGURATION: FILL THESE IN! ---

# --- 1. Initialize Firebase Admin ---
try:
    cred = credentials.Certificate(os.getenv('CREDENTIAL_PATH'))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # Create a reference to the *exact* collection your web app uses
    collection_path = f"/artifacts/{APP_ID}/fish_trades"
    trades_ref = db.collection("artifacts").document(APP_ID).collection("fish_trades")
    portfolio_ref = db.collection("artifacts").document(APP_ID).collection("portfolio_history")
    print(f"Successfully connected to Firestore.")
    print(f"Logging trades to collection: {collection_path}")

except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print("Please ensure 'serviceAccountKey.json' is correct and in the same folder.")
    exit()


# --- 2. Initialize Computer Vision (OpenCV) ---
# Install with: pip install opencv-python

# Open a connection to your webcam (0 is usually the default)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Create the Background Subtractor ---
back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

print("\nWebcam opened. Learning the background...")
print("Please keep the camera perfectly still.")
print("Press 'q' in the video window to quit.")

# --- 3. Main Tracking & Trading Loop ---
last_log_time = time.time()
LOG_INTERVAL = 5 # seconds

pending_stock = None       # the stock selected but not yet traded
pending_decision = None    # BUY or SELL
pending_start_time = None  # when the stock was selected
LOG_INTERVAL = 5           # seconds
current_stock = None

font_path = "arial.ttf"  # replace with path to a TTF font you like
font_size = 40
font = ImageFont.truetype(font_path, font_size)
market_open = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Flip the frame horizontally (webcams are often mirrored)
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape
    mid_x = frame_width // 2 # Integer division
    
    # --- Fish Tracking Logic (MODIFIED) ---

    # 1. Apply the background subtractor
    fg_mask = back_sub.apply(frame)

    # --- 🔽 🔽 🔽 START TUNING HERE 🔽 🔽 🔽 ---

    # 2. Clean up the mask to remove noise
    
    # --- PARAMETER 1: Motion Sensitivity (Threshold) ---
    # Lower this value (e.g., 150) if your fish is spotty or faint.
    # Raise this value (e.g., 220) if you see too much background noise.
    MOTION_SENSITIVITY = 220
    ret, thresh_mask = cv2.threshold(fg_mask, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)
    
    KERNEL_SIZE = (2, 2)
    DILATE_ITERATIONS = 3
    ERODE_ITERATIONS = 3
    
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    clean_mask = cv2.dilate(thresh_mask, kernel, iterations=DILATE_ITERATIONS)
    clean_mask = cv2.erode(clean_mask, kernel, iterations=ERODE_ITERATIONS)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_decision = "HOLD"
    fish_x, fish_y = 0, 0
    found_fish = False

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        MIN_FISH_AREA = 500
        
        if cv2.contourArea(c) > MIN_FISH_AREA: 
            found_fish = True
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                fish_x = int(M["m10"] / M["m00"])
                fish_y = int(M["m01"] / M["m00"])

                cv2.circle(frame, (fish_x, fish_y), 20, (0, 255, 0), 3)
                
                if fish_x < mid_x:
                    current_decision = "BUY"
                else:
                    current_decision = "SELL"


    text_width = 100
    x_center = (frame_width - text_width) // 2
    y_position = 50  # same as before, near top

    # Optionally show the waiting stock or "WAITING"
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    shadow_offset = 2  # for shadow effect

    if market_open:
        if pending_start_time:
            cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (255, 255, 255), 2, cv2.LINE_AA)

            buy_text = "BUY"
            bbox = draw.textbbox((0, 0), buy_text, font=font)
            buy_width = bbox[2] - bbox[0]
            x_buy = mid_x // 4
            y_buy = 50

            # Shadow
            draw.text((x_buy + shadow_offset, y_buy + shadow_offset), buy_text, font=font, fill=(0, 0, 0))
            # Main
            draw.text((x_buy, y_buy), buy_text, font=font, fill=(34, 197, 94))  # green

            # --- SELL text (right side) ---
            sell_text = "SELL"
            bbox = draw.textbbox((0, 0), sell_text, font=font)
            sell_width = bbox[2] - bbox[0]
            x_sell = mid_x + mid_x // 4
            y_sell = 50

            
            
            wait_text = f"WAITING: {current_stock}" if current_stock else "WAITING..."
            bbox = draw.textbbox((0, 0), wait_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_center = (frame_width - text_width) // 2
            y_position = 50
            # Shadow
            draw.text((x_center + shadow_offset, y_position + shadow_offset), wait_text, font=font, fill=(0,0,0))
            # Main text
            draw.text((x_center, y_position), wait_text, font=font, fill=(239,68,68))  # red

        # --- DECISION text at bottom left ---
        decision_text = f"DECISION: {current_decision}"
        bbox = draw.textbbox((0,0), decision_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((20 + shadow_offset, frame_height - 50 + shadow_offset), decision_text, font=font, fill=(0,0,0))  # shadow
        draw.text((20, frame_height - 50), decision_text, font=font, fill=(255,255,255))  # main
    else:
            wait_text = f"NOT OPERATING, MARKET IS CLOSED \n GOOD WEEKEND"
            bbox = draw.textbbox((0, 0), wait_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_center = (frame_width - text_width) // 2
            y_position = 50
            # Shadow
            draw.text((x_center + shadow_offset, y_position + shadow_offset), wait_text, font=font, fill=(0,0,0))
            # Main text
            draw.text((x_center, y_position), wait_text, font=font, fill=(239,68,68))  # red

    # --- Convert PIL back to OpenCV ---
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # --- Now show frame in OpenCV ---
    cv2.imshow("Debug Mask - What the Bot Sees", clean_mask)
    cv2.imshow("The Fin-fluencer BOT (Press 'q' to quit)", frame)


# Get current time in EST
    est = pytz.timezone("US/Eastern")
    now_est = datetime.datetime.now(est)
    current_time = now_est.time()
    current_weekday = now_est.weekday()

    # Market hours: 9:30 AM to 4:00 PM ET, Monday to Friday
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)

    if current_weekday < 5 and market_open <= current_time <= market_close:
        market_open = True
        # --- 1. Fish enters BUY/SELL zone ---
        if current_decision != "HOLD":
            if pending_start_time is None:
                pending_start_time = time.time()
                pending_decision = current_decision
                current_stock = random.choice(tickers)
                print(f"[WAITING] {pending_decision} - {current_stock} for {LOG_INTERVAL} seconds before trading")
        else:
            pending_start_time = None
            pending_decision = None
            current_stock = None

        # --- 2. Execute trade after LOG_INTERVAL ---
        if pending_start_time and (time.time() - pending_start_time >= LOG_INTERVAL):
            # Fetch price
            try:
                stock_data = yf.download(tickers=current_stock, period='1d', interval='1m', progress=False)
                last_price = float(stock_data['Close'].iloc[-1]) if not stock_data.empty else None
            except Exception as e:
                last_price = None
                print(f"Error fetching price for {current_stock}: {e}")

            # Log to Firestore
            decision_data = {
                "decision": pending_decision,
                "stock": current_stock,
                "price": last_price,
                "position_x": fish_x,
                "position_y": fish_y,
                "canvas_width": frame_width,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "status": "pending"
            }

            try:
                trades_ref.add(decision_data)
                print(f"[TRADE LOGGED] Decision: {pending_decision} | Stock: {current_stock} | Price: {last_price}")
            except Exception as e:
                print(f"Error writing to Firestore: {e}")

            pending_start_time = None
            pending_decision = None
    else:
        # Outside market hours, reset any pending trades
        pending_start_time = None
        pending_decision = None
        current_stock = None
        market_open = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Quitting... closing windows.")
cap.release()
cv2.destroyAllWindows()