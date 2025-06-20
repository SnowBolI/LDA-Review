import json
import os
from datetime import datetime
import threading

# Thread lock untuk mencegah race condition saat menulis progress
progress_lock = threading.Lock()

def update_progress(percent, desc, app_name=None):
    """Helper function to update progress for specific app"""
    if app_name:
        progress_file = f"progress_{app_name}.json"
    else:
        progress_file = "progress.json"
    
    progress_data = {
        "percent": percent,
        "description": desc,
        "timestamp": datetime.now().isoformat(),
        "app_name": app_name if app_name else "global"
    }

    try:
        with progress_lock:  # Tambahkan thread lock
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)
            print(f"[Progress Update] {app_name}: {percent}% - {desc}")  # Debug log
    except Exception as e:
        print(f"[Progress Error] {app_name}: {e}")

def get_progress(app_name=None):
    """Get current progress for specific app"""
    if app_name:
        progress_file = f"progress_{app_name}.json"
    else:
        progress_file = "progress.json"
    
    try:
        if os.path.exists(progress_file):
            with progress_lock:
                with open(progress_file, "r") as f:
                    return json.load(f)
        return {
            "percent": 0, 
            "description": "Siap untuk training",
            "timestamp": datetime.now().isoformat(),
            "app_name": app_name if app_name else "global"
        }
    except Exception as e:
        print(f"[Progress Read Error] {app_name}: {e}")
        return {
            "percent": 0, 
            "description": "Error reading progress",
            "timestamp": datetime.now().isoformat(),
            "app_name": app_name if app_name else "global"
        }