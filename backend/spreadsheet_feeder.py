import pandas as pd
import time
import requests
import json
import os # To get file modification time

# --- Configuration ---
CSV_FILE_PATH = "live_feed.csv" # Path to your CSV file
BACKEND_URL = "http://127.0.0.1:8000/live-sensor-data" # Backend IP
MACHINE_ID = "MOT-007" # The machine this data belongs to
CHECK_INTERVAL = 5 # How often to check the file (seconds)

print(f"Starting spreadsheet feeder for {CSV_FILE_PATH}...")
last_mod_time = 0
last_row_count = 0

# Ensure the file exists before starting
if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: CSV file not found at {CSV_FILE_PATH}. Please create it.")
    exit()
else:
     # Initial read to get starting point
     try:
          df_initial = pd.read_csv(CSV_FILE_PATH)
          last_row_count = len(df_initial)
          last_mod_time = os.path.getmtime(CSV_FILE_PATH)
          print(f"Initial row count: {last_row_count}")
     except Exception as e:
          print(f"Error reading initial CSV: {e}")
          exit()


while True:
    try:
        current_mod_time = os.path.getmtime(CSV_FILE_PATH)
        
        # Check if file has been modified since last check
        if current_mod_time > last_mod_time:
            print("Detected file change. Reading new data...")
            df = pd.read_csv(CSV_FILE_PATH)
            current_row_count = len(df)

            # If new rows were added
            if current_row_count > last_row_count:
                # Get only the newest row added
                new_row = df.iloc[-1] 
                
                try:
                    temp = new_row['temperature']
                    hum = new_row['humidity']
                    ts = new_row.get('timestamp', time.time()) # Use timestamp or current time

                    data = {
                        "machine_id": MACHINE_ID,
                        "temperature_c": round(float(temp), 2),
                        "humidity_percent": round(float(hum), 2),
                        "timestamp": float(ts)
                    }
                    
                    print(f"Sending new row: Temp={data['temperature_c']}°C, Hum={data['humidity_percent']}%")
                    
                    try:
                        response = requests.post(BACKEND_URL, json=data, timeout=5)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        print(f"Error sending data: {e}")

                except (ValueError, TypeError, KeyError) as e:
                     print(f"Error processing new row: {e}")

                last_row_count = current_row_count # Update row count
            
            last_mod_time = current_mod_time # Update modification time

    except FileNotFoundError:
        print(f"Warning: CSV file seems to be missing. Waiting...")
    except Exception as e:
         print(f"An error occurred: {e}")
         
    # Wait before checking the file again
    time.sleep(CHECK_INTERVAL)