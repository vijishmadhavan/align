import os
import sys
import subprocess

def install_requirements():
    """Install required packages for face alignment node"""
    print("Installing required packages for Face Alignment node...")
    packages = [
        "insightface",  # Use latest version
        "onnxruntime-gpu" if check_gpu() else "onnxruntime",
        "opencv-python",
        "numpy"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

if __name__ == "__main__":
    install_requirements()
    print("Dependencies installed successfully!") 