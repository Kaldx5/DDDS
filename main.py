import subprocess
import sys
import os

def launch_detector():
    script_path = os.path.join("scripts", "drowsiness_detector.py")
    print(f"DEBUG: Full script path → {os.path.abspath(script_path)}")
    print(f"DEBUG: Using Python executable → {sys.executable}")
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print("=== SCRIPT OUTPUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== SCRIPT ERRORS ===")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Detector script failed with return code {e.returncode}")
        print(e.output)
    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    print("Launching Drowsiness Detection System with DEBUG...")
    launch_detector()
