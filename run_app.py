"""
Simple script to run the Streamlit app
"""

import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

