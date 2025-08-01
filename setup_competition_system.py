import os
import sys
import subprocess
import urllib.request

def setup_competition_environment():
    """Setup complete competition environment"""
    
    print("🏆 Setting up IET Competition System...")
    print("="*50)
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: No virtual environment detected")
        print("   Recommended: Create virtual environment first")
    
    # Install requirements
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Package installation failed")
        return False
    
    # Check for dataset files
    print("\n📊 Checking dataset files...")
    required_files = [
        'pune_pipe_leak_research_dataset_metadata.csv',
        'pune_pipe_leak_research_dataset_vibration_data.npy',
        'pune_pipe_leak_research_dataset_ae_data.npy',
        'pune_pipe_leak_research_dataset_labels.npy'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Dataset files missing. Running dataset generator...")
        try:
            subprocess.check_call([sys.executable, "research_dataset_generator.py"])
            print("✅ Dataset generated successfully")
        except subprocess.CalledProcessError:
            print("❌ Dataset generation failed")
            return False
    
    # Test Streamlit
    print("\n🚀 Testing Streamlit installation...")
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} ready")
    except ImportError:
        print("❌ Streamlit import failed")
        return False
    
    print("\n🎉 Competition system setup complete!")
    print("\n🚀 To run the system:")
    print("   streamlit run app.py")
    print("\n🎯 For competition demo:")
    print("   1. Load dataset from 'Dataset Analysis' page")
    print("   2. Train models from Stage 1 and Stage 2 pages")
    print("   3. Use 'Live Monitoring Dashboard' for demonstration")
    print("   4. Show 'Alert System Demo' for real-time alerts")
    print("   5. Reference 'Research Validation' for technical depth")
    
    return True

if __name__ == "__main__":
    setup_competition_environment()
