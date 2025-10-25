import cv2
import numpy as np
import time
import datetime

# --- Firebase Admin Setup ---
# You must install this library: pip install google-cloud-firestore
# You also need the firebase-admin library: pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore

# --- USER CONFIGURATION: FILL THESE IN! ---

# 1. Get this from the web app UI (e.g., "bX2...")
#    This is the user account your bot will write data to.
USER_ID = "PUT_YOUR_USER_ID_HERE" 

# 2. This is likely 'default-app-id' unless you changed it.
APP_ID = "default-app-id"

# 3. This is the file you downloaded from Firebase.
#    It MUST be in the same folder as this script.
SERVICE_ACCOUNT_KEY_PATH = "serviceAccountKey.json"

# --- END OF CONFIGURATION ---


# --- 1. Initialize Firebase Admin ---
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # Create a reference to the *exact* collection your web app uses
    collection_path = f"/artifacts/{APP_ID}/users/{USER_ID}/fish_trades"
    trades_collection_ref = db.collection(collection_path)
    
    print(f"Successfully connected to Firestore.")
    print(f"Logging trades to collection: {collection_path}")

except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print("Please ensure 'serviceAccountKey.json' is correct and in the same folder.")
    exit()


# --- 2. Initialize Computer Vision (OpenCV) ---
# Install with: pip install opencv-python

# Open a connection to your webcam (0 is usually the default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam opened. Looking for the fish...")
print("Press 'q' in the video window to quit.")

# --- Color-tracking settings ---
# This is set to track a bright orange/yellow-ish color.
# You will need to tune this for your fish!
# Use an "HSV color picker" online to find the range.
# H = Hue (color), S = Saturation (richness), V = Value (brightness)
COLOR_LOWER = np.array([5, 150, 150])
COLOR_UPPER = np.array([25, 255, 255])


# --- 3. Main Tracking & Trading Loop ---

# Timer to log decisions every 5 seconds (like the web app)
last_log_time = time.time()
LOG_INTERVAL = 5 # seconds

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
    
    # --- Fish Tracking Logic ---
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a "mask" that only shows pixels in your color range
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)

    # Find "contours" (outlines) of the colored objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_decision = "HOLD"
    fish_x, fish_y = 0, 0
    found_fish = False

    if len(contours) > 0:
        # Find the *largest* contour (assume it's the fish)
        c = max(contours, key=cv2.contourArea)
        
        # Only track if the object is a reasonable size
        if cv2.contourArea(c) > 500: # Tune this size threshold
            found_fish = True
            
            # Get the center of the fish
            M = cv2.moments(c)
            if M["m00"] != 0:
                fish_x = int(M["m10"] / M["m00"])
                fish_y = int(M["m01"] / M["m00"])

                # Draw a circle on the fish
                cv2.circle(frame, (fish_x, fish_y), 20, (0, 255, 0), 3)
                
                # --- Make the Trading Decision ---
                if fish_x < mid_x:
                    current_decision = "BUY"
                else:
                    current_decision = "SELL"


    # --- Draw UI on the video frame ---
    # Draw the dividing line
    cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add zone text
    cv2.putText(frame, "BUY", (mid_x // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (34, 197, 94), 3)
    cv2.putText(frame, "SELL", (mid_x + mid_x // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (239, 68, 68), 3)

    # Display the current decision
    decision_color = (255, 255, 255) # White for HOLD
    if current_decision == "BUY":
        decision_color = (34, 197, 94) # Green
    elif current_decision == "SELL":
        decision_color = (239, 68, 68) # Red
        
    cv2.putText(frame, f"DECISION: {current_decision}", (20, frame_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, decision_color, 2)

    # Display the video feed
    cv2.imshow("The Fin-fluencer BOT (Press 'q' to quit)", frame)


    # --- 4. Log to Firestore (if time) ---
    current_time = time.time()
    if (current_time - last_log_time > LOG_INTERVAL) and current_decision != "HOLD":
        
        print(f"Logging decision: {current_decision}")
        
        decision_data = {
            "decision": current_decision,
            "stock": "FISH_STOCK_v1",
            "position_x": fish_x,
            "position_y": fish_y,
            "canvas_width": frame_width, # Use frame_width from CV
            "timestamp": datetime.datetime.now(datetime.timezone.utc), # Use timezone-aware time
            "status": "pending"
        }
        
        try:
            # Add the new decision document
            trades_collection_ref.add(decision_data)
            last_log_time = current_time # Reset timer
        
        except Exception as e:
            print(f"Error writing to Firestore: {e}")


    # --- 5. Quit Condition ---
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Quitting... closing windows.")
cap.release()
cv2.destroyAllWindows()
