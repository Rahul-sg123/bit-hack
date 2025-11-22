import serial
import requests
import json
import time

# --- CONFIGURATION ---
# CHANGE THIS to your ESP32's port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
SERIAL_PORT = 'COM8' 
BAUD_RATE = 115200
BACKEND_URL = "http://127.0.0.1:8000/live-sensor-data"
MACHINE_ID = "MOT-007" # The ID of the machine this sensor is attached to
# ---------------------

def start_bridge():
    print(f"🔌 Connecting to ESP32 on {SERIAL_PORT}...")
    
    try:
        # Open the serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for connection to stabilize
        print("✅ Connected! Listening for data...")

        while True:
            if ser.in_waiting > 0:
                try:
                    # 1. Read line from ESP32
                    line = ser.readline().decode('utf-8').strip()
                    
                    # 2. Parse JSON
                    # We filter out debug lines that don't start with '{'
                    if line.startswith('{'):
                        data = json.loads(line)
                        
                        # 3. Add Metadata
                        payload = {
                            "machine_id": MACHINE_ID,
                            "temperature_c": data["temperature_c"],
                            "humidity_percent": data["humidity_percent"],
                            "timestamp": time.time()
                        }

                        # 4. Send to Backend
                        print(f"🚀 Sending: {payload}")
                        response = requests.post(BACKEND_URL, json=payload)
                        
                        if response.status_code == 200:
                            print("   -> Success: Data received by Backend")
                        else:
                            print(f"   -> Error: Backend returned {response.status_code}")
                    
                    else:
                        # Print non-JSON lines (debug info from ESP32)
                        print(f"ESP32 Log: {line}")

                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON received: {line}")
                except Exception as e:
                    print(f"❌ Error: {e}")

    except serial.SerialException as e:
        print(f"❌ Could not open serial port {SERIAL_PORT}. Is it correct? Is Arduino IDE open?")
        print(f"Details: {e}")

if __name__ == "__main__":
    start_bridge()