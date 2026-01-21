import subprocess
import time
import sys
import os

def start_services():
    print("🚀 Starting Black Friday Sales Platform...")
    
    # 1. Start the API (FastAPI)
    print("👉 Starting API on http://localhost:8080")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    # Give the API a moment to start
    time.sleep(2)

    # 2. Start the Dashboard (Dash)
    print("👉 Starting Dashboard on http://localhost:8050")
    dashboard_process = subprocess.Popen(
        [sys.executable, "eda.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    print("\n✅ Both services are running.")
    print("- API: http://localhost:8080")
    print("- Dashboard: http://localhost:8050")
    print("- API Docs: http://localhost:8080/docs")
    print("\nPress Ctrl+C to stop both services.\n")

    try:
        # Keep the script running while processes are alive
        while True:
            time.sleep(1)
            if api_process.poll() is not None or dashboard_process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()
        dashboard_process.terminate()
        print("Done.")

if __name__ == "__main__":
    start_services()
