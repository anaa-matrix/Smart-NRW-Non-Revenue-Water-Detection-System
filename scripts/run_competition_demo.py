import streamlit as st
import subprocess
import sys
import os

def run_competition_demo():
    """Run competition demo with error handling"""
    
    print("🏆 Starting IET Competition Demo...")
    
    # Check environment
    required_files = [
        'app.py',
        'pune_pipe_leak_research_dataset_metadata.csv',
        'pune_pipe_leak_research_dataset_vibration_data.npy'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            print("   Run setup_competition_system.py first")
            return False
    
    # Launch Streamlit
    try:
        print("🚀 Launching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_competition_demo()
