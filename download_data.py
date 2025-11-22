# download_data.py (Improved Version)

import requests
import zipfile
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("data")
DATASET_URL = "https://ti.arc.nasa.gov/c/5/"
DATASET_FILENAME = "Battery_RUL.zip"
DATASET_FILEPATH = DATA_PATH / DATASET_FILENAME
# The approximate expected file size in megabytes.
EXPECTED_SIZE_MB = 25 

def download_nasa_battery_data():
    """Downloads and unzips the NASA Battery Dataset with better error checking."""
    print("--- Starting Data Setup ---")
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download the file (if needed) ---
    if DATASET_FILEPATH.is_file():
        print(f"✅ '{DATASET_FILENAME}' already exists. Skipping download.")
    else:
        print(f"Downloading '{DATASET_FILENAME}' from {DATASET_URL}...")
        try:
            with requests.get(DATASET_URL, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(DATASET_FILEPATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("✅ Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Failed to download file: {e}")
            return # Stop execution if download fails

    # --- Step 2: Verify the downloaded file ---
    try:
        file_size_bytes = DATASET_FILEPATH.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if file_size_mb < EXPECTED_SIZE_MB:
            print(f"⚠️ WARNING: File size is {file_size_mb:.2f} MB, which is smaller than expected.")
            print("The download may be corrupt. Deleting it now. Please run the script again.")
            DATASET_FILEPATH.unlink() # Delete the corrupt file
            return
        else:
            print(f"✅ File size verified: {file_size_mb:.2f} MB.")
            
    except FileNotFoundError:
        print("❌ ERROR: Zip file not found after download attempt.")
        return

    # --- Step 3: Unzip the file ---
    print(f"Unzipping '{DATASET_FILENAME}'...")
    try:
        with zipfile.ZipFile(DATASET_FILEPATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
        print("✅ Unzipping complete. You are ready for the next step!")
    except zipfile.BadZipFile:
        print("❌ ERROR: The file is not a valid zip file or is corrupted.")
        print("Please delete 'data/Battery_RUL.zip' and run this script again.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during unzipping: {e}")

if __name__ == "__main__":
    download_nasa_battery_data()