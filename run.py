import subprocess
import sys

if __name__ == "__main__":
    print("Launching Streamlit Web App...")
    # Executing streamlit using the active python interpreter environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"])
